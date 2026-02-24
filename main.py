import inr_sos

analytical_data = inr_sos.load_mat(
    inr_sos.DATA_DIR + "/DL-based-SoS/train_IC_10k_l2rec_l1rec_imcon.mat"
)

print("Keys:", list(analytical_data.keys()))

for k, v in analytical_data.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
              f"min={v.min():.4f}, max={v.max():.4f}")
    else:
        print(f"  {k}: {type(v)}")