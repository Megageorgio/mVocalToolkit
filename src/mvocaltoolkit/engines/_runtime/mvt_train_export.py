"""Write inference packages from selected checkpoints, preserving the original run."""
import json
import os
import shutil
import sys
from pathlib import Path

from mvt_train_common import atomic_torch, chosen


def export(run, source, kind, out, step):
    import torch
    import yaml
    if out.resolve() == run.resolve() or out.resolve() in run.resolve().parents or out.resolve() == source.resolve() or source.resolve() in out.resolve().parents:
        raise ValueError("Choose a separate export folder outside the run root and engine source")
    path = chosen(run,step)
    cfg = yaml.safe_load((run/"config.yaml").read_text(encoding="utf-8"))
    state = torch.load(path,map_location="cpu",weights_only=False)
    step=state.get("step",state.get("global_step",step))
    out.parent.mkdir(parents=True,exist_ok=True)
    if kind == "mcfl_student":
        from mcfl.export import export as native_export
        weights = state["model"]
        if state.get("ema"): weights.update(state["ema"]["shadow"])
        metadata=state["metadata"]
        temporary=run/"export-student.pt"
        atomic_torch({"state_dict":weights,"size":metadata["size"],"model":metadata["model"],"epoch":state["epoch"]},temporary)
        try:
            if out.exists() and not any(out.iterdir()): out.rmdir()
            if out.exists(): raise ValueError("Choose an empty export folder")
            native_export(str(run),str(out),per_lang=2,ckpt=str(temporary))
        finally: temporary.unlink(missing_ok=True)
    else:
        out.mkdir(parents=True,exist_ok=True)
        if kind in {"tifa_aligner","game_notes"}:
            native=run/"native"
            for name in ("config.yaml","vocabulary.json","lang_map.json"):
                if (native/name).is_file():shutil.copy2(native/name,out/name)
            if kind=="tifa_aligner":
                config=yaml.safe_load((out/"config.yaml").read_text(encoding="utf-8"))
                dictionaries=run/"dictionaries"
                if dictionaries.is_dir():shutil.copytree(dictionaries,out/"dictionaries",dirs_exist_ok=True)
                for converter in config.get("inference",{}).get("g2p",{}).get("converters",[]):
                    if converter.get("id")=="dictionary":converter["kwargs"]["dict_path"]="dictionaries/"+Path(converter["kwargs"]["dict_path"]).name
                (out/"config.yaml").write_text(yaml.safe_dump(config,allow_unicode=True),encoding="utf-8")
            from inference.api import load_state_dict_for_inference
            weights=load_state_dict_for_inference(path)
            atomic_torch({"state_dict":weights},out/"model.pt")
            if kind=="game_notes":
                from inference.api import load_inference_model
                from deployment.api import deploy_model
                from deployment.context import export_mode
                target=out/"onnx"
                with export_mode():
                    model,languages=load_inference_model(out/"model.pt")
                    deploy_model(model=model,lang_map=languages,save_dir=target,dynamo=False,opset_version=17)
                for file in target.iterdir():
                    if file.is_file(): shutil.copy2(file,out/file.name)
        elif kind=="sofa_aligner":
            # SOFA stores its architecture/vocabulary in the Lightning hyperparameters.
            inference={k:v for k,v in state.items() if k in {"state_dict","hyper_parameters","pytorch-lightning_version"}}
            atomic_torch(inference,out/"model.ckpt")
            for name in ("vocab.yaml","global_config.yaml"):
                file=run/"raw/binary"/name
                if file.exists():shutil.copy2(file,out/name)
            records=json.loads((run/"records.json").read_text(encoding="utf-8"))
            phones=sorted({p for r in records for _,_,p in r["phones"] if p not in {"SP","AP"}})
            (out/"dict.txt").write_text("\n".join(f"{p}\t{p}" for p in phones)+"\n",encoding="utf-8")
        elif kind=="wfl_asr_teacher":
            config=yaml.safe_load((run/"wfl/config.yaml").read_text(encoding="utf-8"))
            weights=state["model"]
            if state.get("ema"):weights.update(state["ema"]["shadow"])
            atomic_torch({"state_dict":{"model."+k:v for k,v in weights.items()}},out/"best.pt")
            (out/"config.yaml").write_text(yaml.safe_dump(config,allow_unicode=True),encoding="utf-8")
            for name in ("phonemes.txt","langs.txt","lang_phonemes.json","phoneme_merge_map.json"):
                shutil.copy2(run/name,out/name)
            encoder=run/"wfl/encoder"
            if encoder.is_dir():shutil.copytree(encoder,out/"encoder",dirs_exist_ok=True)
        elif kind=="refiner_boundary":
            atomic_torch({"state_dict":state["model"]},out/"model.pt")
            config=yaml.safe_load((run/"refiner.yaml").read_text(encoding="utf-8"));config.pop("training",None)
            (out/"config.yaml").write_text(yaml.safe_dump(config),encoding="utf-8")
            shutil.copy2(run/"phonemes.txt",out/"phonemes.txt")
        elif kind=="qwen_tts_voice":
            from safetensors.torch import save_file
            base=Path(cfg["base_model"])
            # Preserve tokenizer/config resources, not stale sharded base weights or their index.
            for file in base.rglob("*"):
                relative=file.relative_to(base)
                if any(p.startswith(".") for p in relative.parts):continue
                if file.is_file() and (relative.parts[0]=="speech_tokenizer" or file.suffix not in {".safetensors",".bin",".pt",".pth"} and not file.name.endswith(".index.json")):
                    destination=out/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,destination)
            config=json.loads((base/"config.json").read_text(encoding="utf-8"))
            if cfg.get("tokenizer"):
                shutil.copytree(cfg["tokenizer"],out/"speech_tokenizer",dirs_exist_ok=True,ignore=shutil.ignore_patterns(".*"))
            config["tts_model_type"]="custom_voice"
            config["talker_config"].update(spk_id={cfg["speaker_name"]:3000},spk_is_dialect={cfg["speaker_name"]:False})
            (out/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
            weights={k:v.contiguous().cpu() for k,v in state["model"].items() if not k.startswith("speaker_encoder")}
            weights["talker.model.codec_embedding.weight"][3000]=state["metadata"]["speaker_embedding"]
            save_file(weights,str(out/"model.safetensors"))
        elif kind=="voxcpm_voice":
            from safetensors.torch import save_file
            weights={k:v.contiguous().cpu() for k,v in state["model"].items()}
            if cfg["lora"]:
                save_file(weights,str(out/"lora_weights.safetensors"))
                (out/"lora_config.json").write_text(json.dumps({"base_model":cfg.get("base_model_origin",cfg["base_model"]),"lora_config":{
                    "enable_lm":True,"enable_dit":True,"enable_proj":False,"r":cfg["lora_rank"],"alpha":cfg["lora_rank"],"dropout":0.0}},indent=2),encoding="utf-8")
            else:
                save_file(weights,str(out/"model.safetensors"))
                base=Path(cfg["base_model"])
                for file in base.iterdir():
                    if file.is_file() and (file.name.startswith("audiovae") or file.suffix not in {".safetensors",".bin",".pt",".pth"}):shutil.copy2(file,out/file.name)
                for directory in ("audiovae","audio_vae"):
                    if (base/directory).is_dir():shutil.copytree(base/directory,out/directory,dirs_exist_ok=True)
        else:raise ValueError("Unsupported inference export")
    if (run/"validation.json").is_file():shutil.copy2(run/"validation.json",out/"validation.json")
    manifest=json.loads((run/"input-manifest.json").read_text(encoding="utf-8"))
    (out/"training-provenance.json").write_text(json.dumps({"kind":kind,"step":step,"base_model":cfg["base_model"],
        "licenses":sorted({r["license"] for r in manifest["records"]}),"source":os.environ.get("MVT_SOURCE_DIR","")},ensure_ascii=False,indent=2),encoding="utf-8")
