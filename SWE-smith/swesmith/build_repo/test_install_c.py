"""
Purpose: Test out whether a set of installation commands works for a given C repository at a specific commit.

Usage: python -m swesmith.build_repo.try_install_c owner/repo --commit <commit>
"""

import argparse
import os
import subprocess
import json

from swesmith.profiles.c import CProfile


# Default configuration - modify these for different C repositories
DEFAULT_BUILD_CMDS = [
    "mkdir -p build",
    "cd build",
    "cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    "make -j$(nproc)",
    "cd ..",
]

DEFAULT_TEST_CMD = "ctest --test-dir build --output-on-failure"


def _profile_build_cmds(profile: CProfile) -> str | None:
    """
    Convert profile build_cmds to a single shell string for the install script.
    Skip if the profile uses the default build commands to avoid duplication.
    """
    if hasattr(profile, 'build_cmds') and profile.build_cmds == DEFAULT_BUILD_CMDS:
        return None
    if hasattr(profile, 'build_cmds'):
        return " && ".join(profile.build_cmds)
    return None


def cleanup(repo_name: str, build_dir: str | None = None):
    """Clean up cloned repository and build directory."""
    if os.path.exists(repo_name):
        subprocess.run(
            f"rm -rf {repo_name}",
            check=True,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("> Removed repository")
    
    if build_dir and os.path.exists(build_dir):
        subprocess.run(
            f"rm -rf {build_dir}",
            check=True,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("> Removed build directory")


def main(
    repo: str,
    install_script: str | None = None,
    commit: str = "latest",
    no_cleanup: bool = False,
    force: bool = False,
    build_type: str = "Debug",
    smoke_cmd: str | None = None,
    skip_smoke: bool = False,
    cmake_options: str | None = None,
):
    print(f"> Building C project for {repo} at commit {commit or 'latest'}")
    owner, repo_name = repo.split("/")
    
    # Find the matching profile class dynamically, or create a minimal one
    profile_classes = [cls for name, cls in globals().items() 
                    if isinstance(cls, type) and issubclass(cls, CProfile) and cls != CProfile]
    matching = [cls for cls in profile_classes if cls().owner == owner and cls().repo == repo_name]
    p = matching[0]() if matching else type('TempProfile', (CProfile,), {
        'log_parser': lambda self, log: {}
    })()
    p.owner = owner
    p.repo = repo_name

    use_default_commands = install_script is None
    
    if install_script is not None:
        assert os.path.exists(install_script), (
            f"Installation script {install_script} does not exist"
        )
        assert install_script.endswith(".sh"), "Installation script must be a bash script"
        install_script = os.path.abspath(install_script)
    else:
        print("> Using default build commands from script")

    # Set up environment variables
    env = os.environ.copy()
    env["SWESMITH_BUILD_TYPE"] = build_type
    if cmake_options:
        env["SWESMITH_CMAKE_OPTIONS"] = cmake_options
    
    profile_build_cmds = _profile_build_cmds(p)
    if profile_build_cmds:
        env["SWESMITH_PROFILE_BUILD_CMDS"] = profile_build_cmds

    build_info = {
        "owner": owner,
        "repo": repo_name,
        "commit": commit,
        "build_type": build_type,
        "cmake_options": cmake_options,
    }

    try:
        # Clone repository at the specified commit
        if not os.path.exists(repo_name):
            subprocess.run(
                f"git clone https://github.com/{owner}/{repo_name}.git",
                check=True,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        # Get absolute path before changing directories
        base_dir = os.getcwd()
        os.chdir(repo_name)
        
        if commit != "latest":
            subprocess.run(
                f"git checkout {commit}",
                check=True,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            commit = subprocess.check_output(
                "git rev-parse HEAD", shell=True, text=True
            ).strip()
        
        print(f"> Cloned {repo_name} at commit {commit}")
        p.commit = commit
        build_info["commit"] = commit

        # Define output paths
        output_dir = os.path.join(base_dir, "profiles", "c_envs", f"{owner}_{repo_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        build_json_path = os.path.join(output_dir, "build_info.json")
        install_sh_path = os.path.join(output_dir, "install.sh")

        if (
            os.path.exists(build_json_path)
            and not force
            and input(
                f"> Build info {build_json_path} already exists. Do you want to overwrite it? (y/n) "
            )
            != "y"
        ):
            raise Exception("(No Error) Terminating")

        # Run installation
        print("> Installing C project...")
        if use_default_commands:
            # Use default commands directly
            combined_cmd = " && ".join(DEFAULT_BUILD_CMDS)
            subprocess.run(
                combined_cmd,
                check=True,
                shell=True,
                env=env,
            )
        else:
            # Use provided install script
            subprocess.run(
                ["bash", "-lc", f". {install_script}"],
                check=True,
                env=env,
            )
        print("> Successfully installed C project")

        if not skip_smoke:
            resolved_smoke_cmd = smoke_cmd if smoke_cmd else DEFAULT_TEST_CMD
            
            if resolved_smoke_cmd:
                print(f"> Running smoke test: {resolved_smoke_cmd}")
                result = subprocess.run(
                    resolved_smoke_cmd,
                    check=False,
                    shell=True,
                    env=env,
                )
                if result.returncode == 0:
                    print("> Smoke test passed")
                else:
                    print(f"> Smoke test failed with exit code {result.returncode}")
            else:
                print("> Skipping smoke test")

        # Export build information
        os.chdir("..")
        
        # Save build info as JSON
        with open(build_json_path, "w") as f:
            json.dump(build_info, indent=2, fp=f)
        
        # Copy install script with repository-specific commands
        if use_default_commands:
            install_lines = DEFAULT_BUILD_CMDS
        else:
            with open(install_script) as install_f:
                install_lines = [
                    l.strip("\n") for l in install_f.readlines() if len(l.strip()) > 0
                ]

        with open(install_sh_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        "#!/bin/bash\n",
                        f"# Build script for {owner}/{repo_name} at commit {commit}",
                        f"git clone git@github.com:{owner}/{repo_name}.git",
                        f"cd {repo_name}",
                        f"git checkout {commit}",
                    ]
                    + install_lines
                )
                + "\n"
            )
        
        # Make the script executable
        os.chmod(install_sh_path, 0o755)
        
        print(f"> Exported build information to {output_dir}")
        print(f">   - build_info.json: {build_json_path}")
        print(f">   - install.sh: {install_sh_path}")
        
    except Exception as e:
        print(f"> Installation procedure failed: {e}")
    finally:
        if not no_cleanup:
            cleanup(repo_name, "build")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "repo", type=str, help="Repository name in the format of 'owner/repo'"
    )
    parser.add_argument(
        "install_script",
        type=str,
        nargs="?",
        default=None,
        help="Bash script with installation commands (optional, uses DEFAULT_BUILD_CMDS if not provided)",
    )
    parser.add_argument(
        "-c",
        "--commit",
        type=str,
        help="Commit hash to build the image at (default: latest)",
        default="latest",
    )
    parser.add_argument(
        "--no_cleanup",
        action="store_true",
        help="Do not remove the repository and build directory after installation",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite of existing build info file (if it exists)",
    )
    parser.add_argument(
        "--build-type",
        type=str,
        help="CMake build type (default: Debug)",
        default="Debug",
    )
    parser.add_argument(
        "--smoke-cmd",
        type=str,
        help=f"Optional smoke test command to run (default: {DEFAULT_TEST_CMD})",
        default=None,
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip running a smoke test after installation",
    )
    parser.add_argument(
        "--cmake-options",
        type=str,
        help="Additional CMake options to pass to the build (passed to install script)",
        default=None,
    )

    args = parser.parse_args()
    main(**vars(args))
