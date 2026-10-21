import modal
f = modal.Function.from_name("ply", "work")
print("spawned", f.spawn().object_id)
