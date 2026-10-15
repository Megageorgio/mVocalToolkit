"""Small, checked adaptations of the pinned Applio trainer; upstream files stay intact."""
from __future__ import annotations

import ast


def adapt(source: str, run_dir: str) -> str:
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f"Unsupported Applio trainer contract: {old[:70]}")
        source = source.replace(old, new)
    replace('experiment_dir = os.path.join(current_dir, "logs", model_name)', f"experiment_dir = {run_dir!r}")
    replace("        epoch_str += 1", "        epoch_str = _mvt_resume_epoch")
    replace("        global_step = (epoch_str - 1) * len(train_loader)", "        global_step = _mvt_resume_step")
    replace("    except Exception as e:\n        epoch_str = 1", "    except Exception as e:\n        if _mvt_has_resume:\n            raise\n        epoch_str = 1")
    replace("        info = next(iter(train_loader))", "        info = next(iter(train_loader))\n        _mvt_reference(info)")
    replace('    if os.path.isfile(os.path.join("logs", "reference", embedder_name, "feats.npy")):', "    if False:")
    replace('    print(f"Initializing the generator with {spk_dim} speakers.")',
            '    spk_dim = _mvt_speaker_rows or spk_dim\n    print(f"Initializing the generator with {spk_dim} speakers.")')
    replace("        for batch_idx, info in data_iterator:",
            "        for batch_idx, info in data_iterator:\n            if epoch == _mvt_resume_epoch and batch_idx < _mvt_resume_batch:\n                continue")
    replace("            global_step += 1", "            global_step += 1\n            _mvt_batch(locals())")

    class SingleProcess(ast.NodeTransformer):
        function = ""
        def visit_FunctionDef(self, node):
            previous, self.function = self.function, node.name
            self.generic_visit(node)
            self.function = previous
            return node
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "ExponentialLR":
                for keyword in node.keywords:
                    if keyword.arg == "last_epoch":
                        keyword.value = ast.Constant(-1)
            return node
        def visit_Expr(self, node):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                owner, name = call.func.value.id, call.func.attr
                if (owner, name) == ("dist", "init_process_group"):
                    return ast.copy_location(ast.Pass(), node)
                if (owner, name) == ("os", "_exit"):
                    if self.function == "train_and_evaluate":
                        return ast.copy_location(ast.Return(), node)
                    return ast.copy_location(ast.Raise(exc=ast.Call(func=ast.Name(id="RuntimeError", ctx=ast.Load()),
                        args=[ast.Constant("Applio rejected the training data; see train.log")], keywords=[]), cause=None), node)
            return self.generic_visit(node)
    tree = SingleProcess().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
