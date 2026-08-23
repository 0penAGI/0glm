import sys; sys.path.insert(0, "0glm")
import torch, numpy as np, json
import granular_text_field as g

pool = g.load_pool()
clusters = json.load(open(g.CLUSTERS_CACHE))
sems = g.build_semantics(pool)
sp = g.load_sem_projection(); proj = g.project_sems(sems, *sp)
proj_all = np.concatenate([proj[0], proj[1], proj[2]], axis=0)
affect = g.load_affect()
pairs = g.build_training_pairs(pool, clusters, sems=sems, proj_all=proj_all,
                               affect_all=affect)
print(f"CTX ROWS: {len(pairs[0]['ctx'])} x {len(pairs[0]['ctx'][0])}")
FD = g.FEAT_DIM + g.AFFECT_DIM + g.SEM_PROJ_DIM
model = g.MultiNavigator(feat_dim=FD).to(g.DEVICE)
g.train_multi(model, pairs, n_steps=12000)
torch.save({"model_state": model.state_dict(), "feat_dim": FD},
           "0glm/checkpoints/text_navigator_v15_field.pt")
print("SAVED v15_field")
