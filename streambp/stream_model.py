from transformers.modeling_utils import PreTrainedModel
from typing import List, Optional, Tuple, Union, Any, Dict
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from contextlib import contextmanager
import inspect
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import check_backward_validity, _infer_device_type, _get_autocast_kwargs, _get_device_module, get_device_states, detach_variable

# ---------------------------------------------------------------------------
# Transformers version compatibility shims
# ---------------------------------------------------------------------------
try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
except ImportError:
    from typing import TypedDict
    class FlashAttentionKwargs(TypedDict, total=False):
        pass

try:
    from transformers.processing_utils import Unpack
except ImportError:
    try:
        from typing import Unpack  # Python 3.12+
    except ImportError:
        from typing_extensions import Unpack

try:
    from transformers.utils import LossKwargs
except ImportError:
    from typing import TypedDict
    class LossKwargs(TypedDict, total=False):
        pass

try:
    from transformers.models.llama.modeling_llama import repeat_kv, rotate_half
except ImportError:
    # transformers 5.x moved these; provide fallbacks
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

global_dict = {}
stream_buffer = {}

def apply_rotary_pos_emb(states, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    states_embed = (states * cos) + (rotate_half(states) * sin)
    return states_embed

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class CheckpointFunctionForStreamBackward(torch.autograd.Function):
    chunk_size: int = 100
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.device = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device
        )
        ctx.chunk_size = CheckpointFunctionForStreamBackward.chunk_size
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors
        device_module = _get_device_module(ctx.device)

        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        detached_inputs = detach_variable(tuple(inputs))

        hidden_states_grad = args[0]
        num_chunks = math.ceil(hidden_states_grad.size(1) / ctx.chunk_size)

        if "zero2_optimizer" in global_dict:
            global_dict["zero2_optimizer"].process_gradients = lambda *args, **kwargs: None

        for i in range(num_chunks):
            start = i * ctx.chunk_size
            end = min((i+1)*ctx.chunk_size, hidden_states_grad.size(1))

            if (i == num_chunks - 1) and "zero2_optimizer" in global_dict:
                global_dict["zero2_optimizer"].process_gradients = global_dict["zero2_gradient_process_func"]
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs, chunk_range=(start, end))
                if isinstance(outputs, tuple):
                    hidden_states = outputs[0]
                else:
                    hidden_states = outputs
                torch.autograd.backward(
                        hidden_states,
                        grad_tensors=hidden_states_grad[:, start:end, :].detach(),
                        retain_graph=True if end < hidden_states_grad.size(1) else False
                    )

        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )

        return (None, None) + grads

class StreamMLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.mlp, name)

    def forward(self, x, chunk_range=None):
        if chunk_range is not None:
            x = x[:, chunk_range[0]:chunk_range[1], :]
        down_proj = self.mlp(x)
        return down_proj

# ===========================================================================
# Llama / Qwen2 / Qwen3 style (original StreamAttention + StreamDecoderLayer)
# ===========================================================================

class StreamDecoderLayer(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self._setup_attn()

    def _setup_attn(self):
        self.base_layer.self_attn = StreamAttention(self.base_layer.self_attn)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        chunk_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states: input of shape (batch, seq_len, embed_dim)
            chunk_range: (start, end) indices for query computation only
        """

        residual = hidden_states

        if chunk_range is not None:
            residual = hidden_states[:, chunk_range[0]:chunk_range[1], :]

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            chunk_range=chunk_range,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class StreamAttention(torch.nn.Module):

    def __init__(self, self_attn):
        super().__init__()
        self.self_attn = self_attn
        self.cache_states = {}
        self._setup_stream_buffer()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.self_attn, name)

    def _setup_stream_buffer(self):
        for model in stream_buffer:
            if any(m is self.self_attn for m in model.modules()):
                self.stream_buffer = stream_buffer[model]
                return

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        chunk_range: Optional[Tuple[int, int]] = None,
        key_states: Optional[torch.Tensor] = None,
        value_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        if chunk_range is not None:
            chunk_startidx, chunk_endidx = chunk_range
            chunk_len = chunk_endidx - chunk_startidx
        else:
            chunk_startidx, chunk_endidx, chunk_len = 0, q_len, q_len

        key_states = self.k_proj(hidden_states[:, :chunk_endidx, :])
        value_states = self.v_proj(hidden_states[:, :chunk_endidx, :])
        query_states = self.q_proj(hidden_states[:, chunk_startidx:chunk_endidx, :])

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states = query_states.view(bsz, chunk_len, -1, self.head_dim).transpose(1, 2)

        key_states = key_states.view(bsz, chunk_endidx, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, chunk_endidx, -1, self.head_dim).transpose(1, 2)

        if hasattr(self, "q_norm"):
            query_states = self.q_norm(query_states)
        if hasattr(self, "k_norm"):
            key_states = self.k_norm(key_states)

        key_states = apply_rotary_pos_emb(key_states, cos[:, :chunk_endidx], sin[:, :chunk_endidx])
        query_states = apply_rotary_pos_emb(query_states, cos[:, chunk_startidx:chunk_endidx], sin[:, chunk_startidx:chunk_endidx])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        if query_states.shape[2] == 1 and causal_mask is None:
            is_causal = False
        elif self.stream_buffer["attention_mask"] is None:
            causal_mask = None
            is_causal = True
        else:
            causal_mask = self._generate_causal_mask(chunk_startidx, chunk_endidx, query_states.dtype, query_states.device)
            is_causal = False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            scale=self.scaling,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, chunk_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def _generate_causal_mask(self, start_idx, end_idx, dtype, device):
        sub_attention_mask = self.stream_buffer["attention_mask"][:, :end_idx]
        batch_size = sub_attention_mask.shape[0]

        min_dtype = torch.finfo(dtype).min
        chunk_len = end_idx - start_idx
        causal_mask = torch.full((batch_size, 1, chunk_len, end_idx), fill_value=min_dtype, dtype=dtype, device=device)
        active_mask = torch.arange(start_idx, end_idx, device=device).view(-1, 1) >= torch.arange(end_idx, device=device)
        causal_mask.masked_fill_(active_mask, 0.)

        zero_mask_indices = (sub_attention_mask == 0).unsqueeze(1).unsqueeze(1)
        causal_mask.masked_fill_(zero_mask_indices, min_dtype)

        return causal_mask


# ===========================================================================
# Qwen3_5 Support
# ===========================================================================

class StreamAttentionQwen3_5(torch.nn.Module):
    """
    StreamBP-aware attention for Qwen3_5Attention.

    Key differences vs Llama StreamAttention:
      * q_proj outputs 2 * head_dim per head (query + gate), split before norm.
      * q_norm / k_norm are applied after reshape.
      * attn_output is gated via sigmoid(gate) before o_proj.
      * position_embeddings (cos, sin) shape is (bs, seq_len, head_dim).
      * RoPE applies to both Q and K via a 2-tensor apply_rotary_pos_emb.
      * Uses num_key_value_groups for GQA repeat.
    """

    def __init__(self, self_attn):
        super().__init__()
        self.self_attn = self_attn
        self._setup_stream_buffer()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.self_attn, name)

    def _setup_stream_buffer(self):
        for model in stream_buffer:
            if any(m is self.self_attn for m in model.modules()):
                self.stream_buffer = stream_buffer[model]
                return

    def _qwen3_5_apply_rotary_pos_emb(self, q, k, cos, sin):
        """Qwen3_5-style RoPE with partial rotation (rotary_dim may differ from head_dim)."""
        cos = cos.unsqueeze(1)  # (bs, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1)
        rotary_dim = cos.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
        q_embed = torch.cat([q_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_embed, k_pass], dim=-1)
        return q_embed, k_embed

    def _generate_causal_mask(self, start_idx, end_idx, dtype, device):
        sub_attention_mask = self.stream_buffer["attention_mask"][:, :end_idx]
        batch_size = sub_attention_mask.shape[0]
        min_dtype = torch.finfo(dtype).min
        chunk_len = end_idx - start_idx
        causal_mask = torch.full((batch_size, 1, chunk_len, end_idx), fill_value=min_dtype, dtype=dtype, device=device)
        active_mask = torch.arange(start_idx, end_idx, device=device).view(-1, 1) >= torch.arange(end_idx, device=device)
        causal_mask.masked_fill_(active_mask, 0.)
        zero_mask_indices = (sub_attention_mask == 0).unsqueeze(1).unsqueeze(1)
        causal_mask.masked_fill_(zero_mask_indices, min_dtype)
        return causal_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        chunk_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        bsz, q_len, _ = hidden_states.size()

        if chunk_range is not None:
            chunk_startidx, chunk_endidx = chunk_range
            chunk_len = chunk_endidx - chunk_startidx
        else:
            chunk_startidx, chunk_endidx, chunk_len = 0, q_len, q_len

        head_dim = self.self_attn.head_dim

        # ---- Compute K, V for prefix [:chunk_endidx] ----
        hidden_kv = hidden_states[:, :chunk_endidx, :]
        hidden_shape_kv = (bsz, chunk_endidx, -1, head_dim)

        key_states = self.self_attn.k_norm(
            self.self_attn.k_proj(hidden_kv).view(hidden_shape_kv)
        ).transpose(1, 2)  # (bsz, num_kv_heads, chunk_endidx, head_dim)

        value_states = self.self_attn.v_proj(hidden_kv).view(hidden_shape_kv).transpose(1, 2)

        # ---- Compute Q and gate for current chunk [chunk_startidx:chunk_endidx] ----
        hidden_q = hidden_states[:, chunk_startidx:chunk_endidx, :]
        # q_proj outputs num_heads * head_dim * 2 (query + gate interleaved per head)
        qg_out = self.self_attn.q_proj(hidden_q).view(bsz, chunk_len, -1, head_dim * 2)
        query_states_raw, gate = torch.chunk(qg_out, 2, dim=-1)  # each (bsz, chunk_len, num_heads, head_dim)
        gate = gate.reshape(bsz, chunk_len, -1)  # (bsz, chunk_len, num_heads * head_dim)

        query_states = self.self_attn.q_norm(query_states_raw).transpose(1, 2)  # (bsz, num_heads, chunk_len, head_dim)

        # ---- RoPE ----
        cos, sin = position_embeddings  # (bsz, seq_len, head_dim)
        cos_q = cos[:, chunk_startidx:chunk_endidx, :]   # (bsz, chunk_len, head_dim)
        sin_q = sin[:, chunk_startidx:chunk_endidx, :]
        cos_k = cos[:, :chunk_endidx, :]                 # (bsz, chunk_endidx, head_dim)
        sin_k = sin[:, :chunk_endidx, :]

        query_states, key_states = self._qwen3_5_apply_rotary_pos_emb(
            query_states, key_states, cos_q, sin_q
        )
        # Re-apply RoPE for keys using key-specific positions
        _, key_states = self._qwen3_5_apply_rotary_pos_emb(
            query_states, key_states, cos_k, sin_k
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.self_attn.layer_idx)

        # GQA expansion
        key_states = repeat_kv(key_states, self.self_attn.num_key_value_groups)
        value_states = repeat_kv(value_states, self.self_attn.num_key_value_groups)

        # ---- Build causal mask ----
        if hasattr(self, "stream_buffer") and self.stream_buffer.get("attention_mask") is None:
            causal_mask = None
            is_causal = True
        else:
            causal_mask = self._generate_causal_mask(
                chunk_startidx, chunk_endidx, query_states.dtype, query_states.device
            )
            is_causal = False

        # Contiguous for SDPA
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            scale=self.self_attn.scaling,
            dropout_p=self.self_attn.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # (bsz, num_heads, chunk_len, head_dim) -> (bsz, chunk_len, num_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, chunk_len, -1)

        # Apply gating
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.self_attn.o_proj(attn_output)

        return attn_output, None


class StreamDecoderLayerQwen3_5(nn.Module):
    """
    StreamBP-aware decoder layer for Qwen3_5DecoderLayer.

    Layers with layer_type == "full_attention" have their self_attn replaced
    with StreamAttentionQwen3_5.  Layers with layer_type == "linear_attention"
    are forwarded as-is for the chunk (no streaming through GatedDeltaNet).
    """

    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self.layer_type = base_layer.layer_type
        if self.layer_type == "full_attention":
            self.base_layer.self_attn = StreamAttentionQwen3_5(self.base_layer.self_attn)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        chunk_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> torch.FloatTensor:

        residual = hidden_states
        if chunk_range is not None:
            residual = hidden_states[:, chunk_range[0]:chunk_range[1], :]

        hidden_states = self.base_layer.input_layernorm(hidden_states)

        if self.layer_type == "full_attention":
            hidden_states, _ = self.base_layer.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                chunk_range=chunk_range,
                **kwargs,
            )
        elif self.layer_type == "linear_attention":
            # Linear attention layers don't benefit from StreamBP chunking.
            # Slice to the chunk range and run normally.
            if chunk_range is not None:
                hidden_states = hidden_states[:, chunk_range[0]:chunk_range[1], :]
            hidden_states = self.base_layer.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.base_layer.post_attention_layernorm(hidden_states)
        hidden_states = self.base_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class StreamModelForQwen3_5(PreTrainedModel):
    """
    StreamModel variant for Qwen3_5ForCausalLM.

    The model has a mixed-layer architecture:
      * full_attention layers  -> wrapped with StreamDecoderLayerQwen3_5
      * linear_attention layers -> wrapped but passthrough (no chunked streaming)

    Usage::

        base = Qwen3_5ForCausalLM.from_pretrained(...)
        stream = StreamModelForQwen3_5(
            base,
            gradient_accumulation_steps=1,
            logits_chunk_size=100,
            checkpoint_chunk_size=500,
            stream_checkpoint=True,
        )
        stream.gradient_checkpointing_enable()
    """

    def __init__(
        self,
        model: PreTrainedModel,
        gradient_accumulation_steps: int,
        gradient_accumulation_mode: str = "sum",
        logits_chunk_size: int = 500,
        stream_checkpoint: bool = True,
        checkpoint_chunk_size: int = 500,
    ):
        torch.nn.Module.__init__(self)
        self.supports_gradient_checkpointing = True
        self.logits_chunk_size = logits_chunk_size
        self.stream_checkpoint = stream_checkpoint
        self.checkpoint_chunk_size = checkpoint_chunk_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_accumulation_mode = gradient_accumulation_mode
        self.model = model

        self._setup_stream_buffer()
        self._setup_stream_forward()
        self._setup_gradient_accumulation()

    def _get_text_model(self):
        """
        Traverse PEFT / other wrappers to find the Qwen3_5TextModel that owns .layers.
        Works for both raw Qwen3_5ForCausalLM and PeftModel(Qwen3_5ForCausalLM).
        """
        m = self.model
        # Unwrap PeftModel / LoraModel layers until we reach the causal LM
        for _ in range(5):
            if hasattr(m, "layers"):
                return m          # already at Qwen3_5TextModel
            if hasattr(m, "model"):
                m = m.model
            elif hasattr(m, "base_model"):
                m = m.base_model
            else:
                break
        raise RuntimeError(
            f"Could not find Qwen3_5TextModel with .layers in model tree. "
            f"Final node: {type(m)}"
        )

    def _setup_stream_buffer(self):
        if self not in stream_buffer:
            stream_buffer[self] = {}
        self.stream_buffer = stream_buffer[self]

        text_model = self._get_text_model()
        text_model.stream_buffer = self.stream_buffer

        def _attention_mask_recording_wrapper(func, *args, **kwargs):
            def wrapped_func(*args, **kwargs):
                if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                    self.stream_buffer["attention_mask"] = kwargs["attention_mask"]
                return func(*args, **kwargs)
            return wrapped_func

        text_model.forward = _attention_mask_recording_wrapper(text_model.forward)

    def _setup_stream_forward(self):
        layers = self._get_text_model().layers
        for i in range(len(layers)):
            layers[i] = StreamDecoderLayerQwen3_5(layers[i])

    def _setup_gradient_accumulation(self):
        self._cur_gradient_accumulation_step = 0
        self._valid_pos_num = 0

    @contextmanager
    def original_model_context(self):
        text_model = self._get_text_model()
        original_stream_layers = []
        original_stream_attentions = []

        for i, stream_layer in enumerate(text_model.layers):
            original_stream_layers.append(stream_layer)
            if stream_layer.layer_type == "full_attention":
                original_stream_attentions.append(stream_layer.base_layer.self_attn)
                stream_layer.base_layer.self_attn = stream_layer.base_layer.self_attn.self_attn
            else:
                original_stream_attentions.append(None)
            text_model.layers[i] = stream_layer.base_layer

        try:
            yield self.model
        finally:
            for i, (stream_layer, stream_attention) in enumerate(
                zip(original_stream_layers, original_stream_attentions)
            ):
                if stream_layer.layer_type == "full_attention" and stream_attention is not None:
                    stream_layer.base_layer.self_attn = stream_attention
                text_model.layers[i] = stream_layer

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        with self.original_model_context() as original_model:
            return original_model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        with self.original_model_context() as original_model:
            return original_model.load_state_dict(state_dict, strict=strict, assign=assign)

    def save_pretrained(self, save_directory, **kwargs):
        with self.original_model_context() as original_model:
            return original_model.save_pretrained(save_directory, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def gradient_checkpointing_enable(
        self: "PreTrainedModel",
        gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None,
    ):
        from functools import partial

        if not self.supports_gradient_checkpointing:
            raise ValueError("{} does not support gradient checkpointing.".format(self.__class__.__name__))

        if self.stream_checkpoint:
            def stream_gradient_checkpointing_func(func, *args, **kwargs):
                CheckpointFunctionForStreamBackward.chunk_size = self.checkpoint_chunk_size
                preserve = kwargs.pop("preserve_rng_state", True)
                return CheckpointFunctionForStreamBackward.apply(func, preserve, *args, **kwargs)
            gradient_checkpointing_func = stream_gradient_checkpointing_func
        else:
            if gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {"use_reentrant": True}
            from torch.utils.checkpoint import checkpoint
            gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)

        def custom_gradient_checkpointing_func(func, *args, **kwargs):
            if hasattr(func, "func"):
                func = func.func
            module: "torch.nn.Module" = func.__self__

            if any(param.requires_grad for param in module.parameters()):
                for arg in args:
                    if torch.is_tensor(arg) and torch.is_floating_point(arg):
                        arg.requires_grad_(True)
                        break

            return gradient_checkpointing_func(func, *args, **kwargs)

        if "value" in inspect.signature(self._set_gradient_checkpointing).parameters:
            self.apply(partial(self._set_gradient_checkpointing, value=True))
            self.enable_input_require_grads()
        else:
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=custom_gradient_checkpointing_func)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if "zero2_optimizer" in global_dict:
            global_dict["zero2_gradient_process_func"] = global_dict["zero2_optimizer"].process_gradients
            global_dict["zero2_optimizer"].process_gradients = lambda *args, **kwargs: None

        if attention_mask is not None:
            self.stream_buffer["attention_mask"] = attention_mask
            attention_mask = None

        if (not self.training) or (not torch.is_grad_enabled()):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Resolve the actual Qwen3_5ForCausalLM regardless of PEFT wrapping
        # PeftModel → LoraModel → Qwen3_5ForCausalLM
        causal_model = self.model
        while not hasattr(causal_model, "lm_head"):
            causal_model = causal_model.model if hasattr(causal_model, "model") else causal_model.base_model

        text_model = self._get_text_model()  # Qwen3_5TextModel that owns .layers

        # Forward through Qwen3_5TextModel
        outputs = text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs[0]
        B, T, C = hidden_states.size()
        batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)

        loss = torch.tensor(0., device=hidden_states.device)
        num_chunks = math.ceil(T / self.logits_chunk_size)

        detached_hidden_states = hidden_states.detach().contiguous().requires_grad_(True)

        if causal_model.lm_head.weight.grad is None:
            causal_model.lm_head.weight.grad = torch.zeros_like(causal_model.lm_head.weight)

        for i in range(num_chunks):
            start = i * self.logits_chunk_size
            end = min((i + 1) * self.logits_chunk_size + 1, T)

            logits_chunk = causal_model.lm_head(detached_hidden_states[:, start:end, :])
            labels_chunk = labels[:, start:end]

            chunk_valid_posnum = (labels_chunk != -100).sum().item() - batch_size

            if chunk_valid_posnum <= 0:
                continue

            loss_chunk = causal_model.loss_function(
                logits=logits_chunk, labels=labels_chunk, vocab_size=causal_model.config.vocab_size
            ) * chunk_valid_posnum
            loss_chunk.backward()
            del logits_chunk
            loss += loss_chunk.detach()

        batch_valid_posnum = (labels != -100).sum().item()
        loss.div_(batch_valid_posnum)

        torch.autograd.backward(hidden_states, grad_tensors=detached_hidden_states.grad.detach())
        detached_hidden_states.grad = None

        self._cur_gradient_accumulation_step += 1
        self._valid_pos_num += batch_valid_posnum
        if self._cur_gradient_accumulation_step == self.gradient_accumulation_steps:
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.div_(self._valid_pos_num)
            self._cur_gradient_accumulation_step = 0
            self._valid_pos_num = 0

        if not return_dict:
            return (loss,) + (None,) + outputs[1:]

        return CausalLMOutputWithPast(
            loss=loss,
            past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )


# ===========================================================================
# OpenAI GPT Support
# ===========================================================================

class StreamAttentionGPT(torch.nn.Module):
    """
    StreamBP-aware attention for the original OpenAI GPT Attention module.

    Key differences:
      * No RoPE; positional information comes from absolute embeddings upstream.
      * QKV are computed jointly via a single Conv1D (c_attn), then split.
      * Causal mask is based on a fixed lower-triangular bias tensor.
      * Output projection is c_proj.
      * n_head-based split/merge heads.

    For StreamBP, we:
      1. Run c_attn on hidden_states[:, :chunk_endidx, :] to get K, V
         (and a "wasted" full Q which we discard outside the chunk range).
      2. Slice Q for [chunk_startidx:chunk_endidx].
      3. Run attention over the chunk rows against all prefix columns.
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self._setup_stream_buffer()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.attn, name)

    def _setup_stream_buffer(self):
        for model in stream_buffer:
            if any(m is self.attn for m in model.modules()):
                self.stream_buffer = stream_buffer[model]
                return

    def _split_heads_range(self, x, start, end, k=False):
        """Split x[:, start:end, :] into multi-head format."""
        chunk = x[:, start:end, :]
        bsz, chunk_len, _ = chunk.size()
        new_shape = (bsz, chunk_len, self.attn.n_head, chunk.size(-1) // self.attn.n_head)
        chunk = chunk.view(*new_shape)
        if k:
            return chunk.permute(0, 2, 3, 1)  # (bsz, n_head, head_dim, seq_len)
        else:
            return chunk.permute(0, 2, 1, 3)  # (bsz, n_head, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        chunk_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = x.size()

        if chunk_range is not None:
            chunk_startidx, chunk_endidx = chunk_range
            chunk_len = chunk_endidx - chunk_startidx
        else:
            chunk_startidx, chunk_endidx, chunk_len = 0, q_len, q_len

        # Run c_attn on the prefix up to chunk_endidx
        prefix = x[:, :chunk_endidx, :]
        all_qkv = self.attn.c_attn(prefix)                       # (bsz, chunk_endidx, 3*n_state)
        query_full, key_full, value_full = all_qkv.split(self.attn.split_size, dim=2)

        # Q: only the chunk rows
        query = self._split_heads_range(query_full, chunk_startidx, chunk_endidx)
        # K, V: all rows in prefix
        key   = self._split_heads_range(key_full,   0, chunk_endidx, k=True)
        value = self._split_heads_range(value_full, 0, chunk_endidx)

        # Attention scores: (bsz, n_head, chunk_len, chunk_endidx)
        w = torch.matmul(query, key)  # Q * K^T
        if self.attn.scale:
            w = w / math.sqrt(value.size(-1))

        # Causal bias: lower-triangular
        # self.attn.bias shape: (1, 1, n_positions, n_positions)
        b = self.attn.bias[:, :, chunk_startidx:chunk_endidx, :chunk_endidx]
        w = w * b + -1e4 * (1 - b)

        if attention_mask is not None:
            # attention_mask from GPT is already pre-processed to additive form
            # but in StreamBP we bypass the outer preprocessing, so handle raw
            if self.stream_buffer.get("attention_mask") is not None:
                raw_mask = self.stream_buffer["attention_mask"][:, :chunk_endidx]
                # additive mask: 0 for attend, -large for ignore
                raw_mask_float = (1.0 - raw_mask.unsqueeze(1).unsqueeze(1).float()) * torch.finfo(w.dtype).min
                w = w + raw_mask_float
            else:
                w = w + attention_mask

        w = nn.functional.softmax(w, dim=-1)
        w = self.attn.attn_dropout(w)

        a = torch.matmul(w, value)  # (bsz, n_head, chunk_len, head_dim)
        a = a.permute(0, 2, 1, 3).contiguous()
        bsz2, chunk_len2, n_head, head_dim = a.size()
        a = a.view(bsz2, chunk_len2, n_head * head_dim)

        a = self.attn.c_proj(a)
        a = self.attn.resid_dropout(a)

        outputs = [a]
        if output_attentions:
            outputs.append(w)
        return outputs


class StreamDecoderLayerGPT(nn.Module):
    """
    StreamBP-aware wrapper for OpenAI GPT Block.

    A GPT Block has the structure:
        ln_1 -> attn -> residual -> ln_2 -> mlp -> residual
    (note: ln is applied BEFORE the sub-layer, not after).
    """

    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.attn = StreamAttentionGPT(self.base_layer.attn)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        chunk_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        # Residual for the chunk only
        if chunk_range is not None:
            residual_attn = x[:, chunk_range[0]:chunk_range[1], :]
        else:
            residual_attn = x

        # Attention sub-layer (ln_1 applied to full x, then chunked attention)
        x_normed = self.base_layer.ln_1(x)
        attn_outputs = self.base_layer.attn(
            x_normed,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            chunk_range=chunk_range,
            **kwargs,
        )
        a = attn_outputs[0]  # (bsz, chunk_len, n_state)

        n = residual_attn + a  # (bsz, chunk_len, n_state)

        # MLP sub-layer
        n_normed = self.base_layer.ln_2(n)
        m = self.base_layer.mlp(n_normed)
        h = n + m

        outputs = [h] + attn_outputs[1:]
        return outputs


class StreamModelForGPT(PreTrainedModel):
    """
    StreamModel variant for OpenAIGPTLMHeadModel.

    Notable structural differences from Llama:
      * Inner backbone at model.transformer (not model.model).
      * Layers at model.transformer.h (not model.model.layers).
      * lm_head weights are tied to model.transformer.tokens_embed.
      * No RoPE; absolute positional embeddings computed inside the backbone.

    Usage::

        base = OpenAIGPTLMHeadModel.from_pretrained(...)
        stream = StreamModelForGPT(
            base,
            gradient_accumulation_steps=1,
            logits_chunk_size=100,
            checkpoint_chunk_size=100,
            stream_checkpoint=True,
        )
        stream.gradient_checkpointing_enable()
    """

    def __init__(
        self,
        model: PreTrainedModel,
        gradient_accumulation_steps: int,
        gradient_accumulation_mode: str = "sum",
        logits_chunk_size: int = 500,
        stream_checkpoint: bool = True,
        checkpoint_chunk_size: int = 500,
    ):
        torch.nn.Module.__init__(self)
        self.supports_gradient_checkpointing = True
        self.logits_chunk_size = logits_chunk_size
        self.stream_checkpoint = stream_checkpoint
        self.checkpoint_chunk_size = checkpoint_chunk_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_accumulation_mode = gradient_accumulation_mode
        self.model = model

        self._setup_stream_buffer()
        self._setup_stream_forward()
        self._setup_gradient_accumulation()

    def _setup_stream_buffer(self):
        if self not in stream_buffer:
            stream_buffer[self] = {}
        self.stream_buffer = stream_buffer[self]

        # For GPT the backbone is model.transformer
        base_model = self.model.transformer
        base_model.stream_buffer = self.stream_buffer

        def _attention_mask_recording_wrapper(func, *args, **kwargs):
            def wrapped_func(*args, **kwargs):
                if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                    self.stream_buffer["attention_mask"] = kwargs["attention_mask"]
                return func(*args, **kwargs)
            return wrapped_func

        base_model.forward = _attention_mask_recording_wrapper(base_model.forward)

    def _setup_stream_forward(self):
        # GPT blocks are at model.transformer.h
        for i in range(len(self.model.transformer.h)):
            self.model.transformer.h[i] = StreamDecoderLayerGPT(self.model.transformer.h[i])

    def _setup_gradient_accumulation(self):
        self._cur_gradient_accumulation_step = 0
        self._valid_pos_num = 0

    @contextmanager
    def original_model_context(self):
        original_stream_layers = []
        original_stream_attentions = []

        for i, stream_layer in enumerate(self.model.transformer.h):
            original_stream_layers.append(stream_layer)
            original_stream_attentions.append(stream_layer.base_layer.attn)
            stream_layer.base_layer.attn = stream_layer.base_layer.attn.attn  # unwrap
            self.model.transformer.h[i] = stream_layer.base_layer

        try:
            yield self.model
        finally:
            for i, (stream_layer, stream_attn) in enumerate(
                zip(original_stream_layers, original_stream_attentions)
            ):
                stream_layer.base_layer.attn = stream_attn
                self.model.transformer.h[i] = stream_layer

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        with self.original_model_context() as original_model:
            return original_model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        with self.original_model_context() as original_model:
            return original_model.load_state_dict(state_dict, strict=strict, assign=assign)

    def save_pretrained(self, save_directory, **kwargs):
        with self.original_model_context() as original_model:
            return original_model.save_pretrained(save_directory, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def gradient_checkpointing_enable(
        self: "PreTrainedModel",
        gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None,
    ):
        from functools import partial

        if not self.supports_gradient_checkpointing:
            raise ValueError("{} does not support gradient checkpointing.".format(self.__class__.__name__))

        if self.stream_checkpoint:
            def stream_gradient_checkpointing_func(func, *args, **kwargs):
                CheckpointFunctionForStreamBackward.chunk_size = self.checkpoint_chunk_size
                preserve = kwargs.pop("preserve_rng_state", True)
                return CheckpointFunctionForStreamBackward.apply(func, preserve, *args, **kwargs)
            gradient_checkpointing_func = stream_gradient_checkpointing_func
        else:
            if gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {"use_reentrant": True}
            from torch.utils.checkpoint import checkpoint
            gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)

        def custom_gradient_checkpointing_func(func, *args, **kwargs):
            if hasattr(func, "func"):
                func = func.func
            module: "torch.nn.Module" = func.__self__

            if any(param.requires_grad for param in module.parameters()):
                for arg in args:
                    if torch.is_tensor(arg) and torch.is_floating_point(arg):
                        arg.requires_grad_(True)
                        break

            return gradient_checkpointing_func(func, *args, **kwargs)

        if "value" in inspect.signature(self._set_gradient_checkpointing).parameters:
            self.apply(partial(self._set_gradient_checkpointing, value=True))
            self.enable_input_require_grads()
        else:
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=custom_gradient_checkpointing_func)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if "zero2_optimizer" in global_dict:
            global_dict["zero2_gradient_process_func"] = global_dict["zero2_optimizer"].process_gradients
            global_dict["zero2_optimizer"].process_gradients = lambda *args, **kwargs: None

        if attention_mask is not None:
            self.stream_buffer["attention_mask"] = attention_mask
            attention_mask = None  # suppress creation of the full T×T mask

        if (not self.training) or (not torch.is_grad_enabled()):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        from transformers.modeling_outputs import CausalLMOutput
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        causal_model = self.model  # OpenAIGPTLMHeadModel

        # Forward through OpenAIGPTModel (backbone)
        transformer_outputs = causal_model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]  # (bsz, seq_len, n_embd)
        B, T, C = hidden_states.size()
        batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)

        loss = torch.tensor(0., device=hidden_states.device)
        num_chunks = math.ceil(T / self.logits_chunk_size)

        detached_hidden_states = hidden_states.detach().contiguous().requires_grad_(True)

        if causal_model.lm_head.weight.grad is None:
            causal_model.lm_head.weight.grad = torch.zeros_like(causal_model.lm_head.weight)

        for i in range(num_chunks):
            start = i * self.logits_chunk_size
            end = min((i + 1) * self.logits_chunk_size + 1, T)

            logits_chunk = causal_model.lm_head(detached_hidden_states[:, start:end, :])
            labels_chunk = labels[:, start:end]

            chunk_valid_posnum = (labels_chunk != -100).sum().item() - batch_size

            if chunk_valid_posnum <= 0:
                continue

            loss_chunk = causal_model.loss_function(
                logits=logits_chunk, labels=labels_chunk, vocab_size=causal_model.config.vocab_size
            ) * chunk_valid_posnum
            loss_chunk.backward()
            del logits_chunk
            loss += loss_chunk.detach()

        batch_valid_posnum = (labels != -100).sum().item()
        loss.div_(batch_valid_posnum)

        torch.autograd.backward(hidden_states, grad_tensors=detached_hidden_states.grad.detach())
        detached_hidden_states.grad = None

        self._cur_gradient_accumulation_step += 1
        self._valid_pos_num += batch_valid_posnum
        if self._cur_gradient_accumulation_step == self.gradient_accumulation_steps:
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.div_(self._valid_pos_num)
            self._cur_gradient_accumulation_step = 0
            self._valid_pos_num = 0

        if not return_dict:
            output = (None,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=None,
            hidden_states=transformer_outputs.hidden_states if hasattr(transformer_outputs, "hidden_states") else None,
            attentions=transformer_outputs.attentions if hasattr(transformer_outputs, "attentions") else None,
        )


# ===========================================================================
# Original StreamModel (Llama / Qwen2 / Qwen3 style)
# ===========================================================================

class StreamModel(PreTrainedModel):
    def __init__(self, model: PreTrainedModel, gradient_accumulation_steps, gradient_accumulation_mode="sum", logits_chunk_size: int=500, stream_checkpoint: bool=True, checkpoint_chunk_size: int=500):
        """ The StreamModel class wraps the original model to save the memory usage. """
        torch.nn.Module.__init__(self)
        self.supports_gradient_checkpointing = True
        self.logits_chunk_size = logits_chunk_size
        self.stream_checkpoint = stream_checkpoint
        self.checkpoint_chunk_size = checkpoint_chunk_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_accumulation_mode = gradient_accumulation_mode
        self.model = model

        self._setup_stream_buffer()
        self._setup_stream_forward()
        self._setup_gradient_accumulation()

    def _setup_stream_buffer(self):
        if self not in stream_buffer:
            stream_buffer[self] = {}
        self.stream_buffer = stream_buffer[self]

        base_model = self.get_base_model(self.model)
        base_model.stream_buffer = self.stream_buffer
        def _attention_mask_recording_wrapper(func, *args, **kwargs):
            def wrapped_func(*args, **kwargs):
                if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                    self.stream_buffer["attention_mask"] = kwargs["attention_mask"]
                return func(*args, **kwargs)
            return wrapped_func
        base_model.forward = _attention_mask_recording_wrapper(base_model.forward)

    def _setup_stream_forward(self):
        for i in range(len(self.model.model.layers)):
            self.model.model.layers[i] = StreamDecoderLayer(self.model.model.layers[i])

    def _setup_gradient_accumulation(self):
        self._cur_gradient_accumulation_step = 0
        self._valid_pos_num = 0

    def get_base_model(self, model):
        is_base_model = False
        while not is_base_model:
            for attr in ["model", "base_model", "module"]:
                if hasattr(model, attr) and not (getattr(model, attr) is model):
                    model = getattr(model, attr)
                else:
                    is_base_model = True
                    break
        return model

    @contextmanager
    def original_model_context(self):
        original_stream_layers = []
        original_stream_attentions = []

        for i, stream_layer in enumerate(self.model.model.layers):
            original_stream_layers.append(stream_layer)
            original_stream_attentions.append(stream_layer.base_layer.self_attn)
            stream_layer.base_layer.self_attn = stream_layer.base_layer.self_attn.self_attn
            self.model.model.layers[i] = stream_layer.base_layer

        try:
            yield self.model
        finally:
            for i, (stream_layer, stream_attention) in enumerate(zip(original_stream_layers, original_stream_attentions)):
                stream_layer.base_layer.self_attn = stream_attention
                self.model.model.layers[i] = stream_layer

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        with self.original_model_context() as original_model:
            return original_model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        with self.original_model_context() as original_model:
            return original_model.load_state_dict(state_dict, strict=strict, assign=assign)

    def save_pretrained(self, save_directory, **kwargs):
        with self.original_model_context() as original_model:
            return original_model.save_pretrained(save_directory, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def gradient_checkpointing_enable(self: "PreTrainedModel", gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None):
        r"""
        Activates gradient checkpointing for the current model.

        Modification of the original method to enable gradient checkpointing for block-wise optimizer.
        """
        from functools import partial

        if not self.supports_gradient_checkpointing:
            raise ValueError("{} does not support gradient checkpointing.".format(self.__class__.__name__))

        if self.stream_checkpoint:
            def stream_gradient_checkpointing_func(func, *args, **kwargs):
                CheckpointFunctionForStreamBackward.chunk_size = self.checkpoint_chunk_size
                preserve = kwargs.pop("preserve_rng_state", True)
                return CheckpointFunctionForStreamBackward.apply(func, preserve, *args, **kwargs)

            gradient_checkpointing_func = stream_gradient_checkpointing_func
        else:
            if gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {"use_reentrant": True}
            from torch.utils.checkpoint import checkpoint
            gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)

        def custom_gradient_checkpointing_func(func, *args, **kwargs):
            if hasattr(func, "func"):
                func = func.func
            module: "torch.nn.Module" = func.__self__

            if any(param.requires_grad for param in module.parameters()):
                for arg in args:
                    if torch.is_tensor(arg) and torch.is_floating_point(arg):
                        arg.requires_grad_(True)
                        break

            return gradient_checkpointing_func(func, *args, **kwargs)

        if "value" in inspect.signature(self._set_gradient_checkpointing).parameters:
            self.apply(partial(self._set_gradient_checkpointing, value=True))
            self.enable_input_require_grads()
        else:
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=custom_gradient_checkpointing_func)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if "zero2_optimizer" in global_dict:
            global_dict["zero2_gradient_process_func"] = global_dict["zero2_optimizer"].process_gradients
            global_dict["zero2_optimizer"].process_gradients = lambda *args, **kwargs: None

        if attention_mask is not None:
            self.stream_buffer["attention_mask"] = attention_mask
            attention_mask = None

        if (not self.training) or (not torch.is_grad_enabled()):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
                **kwargs,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model = self.model

        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        B, T, C = hidden_states.size()
        batch_size = input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)

        loss = torch.tensor(0., device=hidden_states.device)
        num_chunks = math.ceil(T / self.logits_chunk_size)

        detached_hidden_states = hidden_states.detach().contiguous().requires_grad_(True)

        if model.lm_head.weight.grad is None:
            model.lm_head.weight.grad = torch.zeros_like(model.lm_head.weight)

        for i in range(num_chunks):
            start = i * self.logits_chunk_size
            end = min((i+1)*self.logits_chunk_size+1, T)

            logits_chunk = model.lm_head(detached_hidden_states[:, start:end, :])
            labels_chunk = labels[:, start:end]

            chunk_valid_posnum = (labels_chunk != -100).sum().item() - batch_size

            if chunk_valid_posnum <= 0:
                continue

            loss_chunk = model.loss_function(logits=logits_chunk, labels=labels_chunk, vocab_size=model.config.vocab_size) * chunk_valid_posnum
            loss_chunk.backward()

            del logits_chunk
            loss += loss_chunk.detach()

        batch_valid_posnum = (labels != -100).sum().item()
        loss.div_(batch_valid_posnum)

        torch.autograd.backward(hidden_states, grad_tensors=detached_hidden_states.grad.detach())

        detached_hidden_states.grad = None

        self._cur_gradient_accumulation_step += 1
        self._valid_pos_num += batch_valid_posnum
        if self._cur_gradient_accumulation_step == self.gradient_accumulation_steps:
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.div_(self._valid_pos_num)
            self._cur_gradient_accumulation_step = 0
            self._valid_pos_num = 0

        if not return_dict:
            output = (None,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
