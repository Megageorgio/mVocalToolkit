"""Normalize SOFA training bases and transfer matching phoneme rows by name."""
from pathlib import Path


def prepare_base(run,value):
    import torch
    import lightning.pytorch as pl
    root=Path(value)
    if root.is_dir():
        checkpoint=next(iter(root.glob("*.ckpt")),None)
        if checkpoint is None: checkpoint=next(iter(root.glob("*.safetensors")),None)
    else: checkpoint=root;root=root.parent
    if checkpoint is None: raise ValueError("The SOFA base has no checkpoint or safetensors weights")
    if checkpoint.suffix==".safetensors":
        import importlib.util
        file=Path(__file__).resolve().parents[1]/"sofa/worker.py"
        spec=importlib.util.spec_from_file_location("mvt_sofa_base_loader",file)
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
        layout={"checkpoint":checkpoint.name,"format":"safetensors","train_config":"train_config.yaml","global_config":"global_config.yaml","vocab":"vocab.yaml"}
        task=worker.LoadedModel(str(root),layout,"cpu").task
        state={"state_dict":task.state_dict(),"hyper_parameters":dict(task.hparams),"pytorch-lightning_version":pl.__version__}
    else:
        state=torch.load(checkpoint,map_location="cpu",weights_only=False)
        state={key:state[key] for key in ("state_dict","hyper_parameters","pytorch-lightning_version")}
    params=state["hyper_parameters"]
    if not all(k in params for k in ("model_config","melspec_config","vocab_text")):
        raise ValueError("The SOFA base lacks architecture, features or vocabulary")
    from mvt_train_common import atomic_torch
    target=run/"base-sofa.ckpt";atomic_torch(state,target)
    return str(target),params


def transfer(task,base):
    """The upstream loader reuses a head by size alone; equal-size vocabularies may have different IDs."""
    import torch
    task.backbone.load_state_dict(base.backbone.state_dict(),strict=True)
    with torch.no_grad():
        task.head.weight[-2:].copy_(base.head.weight[-2:])
        task.head.bias[-2:].copy_(base.head.bias[-2:])
        for phone,new_id in task.vocab.items():
            old_id=base.vocab.get(phone)
            if not isinstance(phone,str) or phone.startswith("<") or not isinstance(old_id,int) or not isinstance(new_id,int):continue
            if not 0 <= new_id < task.head.out_features-2 or not 0 <= old_id < base.head.out_features-2:continue
            task.head.weight[new_id].copy_(base.head.weight[old_id])
            task.head.bias[new_id].copy_(base.head.bias[old_id])
