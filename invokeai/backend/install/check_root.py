"""
Check that the invokeai_root is correctly configured and exit if not.
"""

import sys

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.frontend.cli.arg_parser import InvokeAIArgs

CORE_MODELS = {
    "CLIP-ViT-bigG-14-laion2B-39B-b160k",  # SDXL Tokenizer 2
    "bert-base-uncased",
    "clip-vit-large-patch14",  # SD-1
    "sd-vae-ft-mse",
    "stable-diffusion-2-clip",
    "stable-diffusion-safety-checker",
}


def get_missing_core_models(config: InvokeAIAppConfig) -> set[str]:
    model_paths = {config.models_path / f"core/convert/{model}" for model in CORE_MODELS}
    return {str(path) for path in model_paths if not path.exists()}


# TODO(psyche): Should this also check for things like ESRGAN models, database, etc?
def validate_root_structure(config: InvokeAIAppConfig) -> None:
    assert config.db_path.parent.exists(), f"{config.db_path.parent} not found"
    assert config.models_path.exists(), f"{config.models_path} not found"
    missing_core_models = get_missing_core_models(config)
    if not config.ignore_missing_core_models and (len(missing_core_models) > 0):
        joined_models = ", ".join(missing_core_models)
        raise Exception(f"Missing core safetensor conversion models: {joined_models}")


def check_invokeai_root(config: InvokeAIAppConfig):
    ignore_missing_core_models = getattr(InvokeAIArgs.args, "ignore_missing_core_models", False)
    if not ignore_missing_core_models:
        try:
            validate_root_structure(config)
        except Exception as e:
            print()
            print(f"An exception has occurred: {str(e)}")
            print("== STARTUP ABORTED ==")
            print("** One or more necessary files is missing from your InvokeAI root directory **")
            print("** Please rerun the configuration script to fix this problem. **")
            print("** From the launcher, selection option [6]. **")
            print(
                '** From the command line, activate the virtual environment and run "invokeai-configure --yes --skip-sd-weights" **'
            )
            print(
                '** (To skip this check completely, add "--ignore_missing_core_models" to your CLI args. Not installing '
                "these core models will prevent the loading of some or all .safetensors and .ckpt files. However, you can "
                "always come back and install these core models in the future.)"
            )
            input("Press any key to continue...")
            sys.exit(0)
