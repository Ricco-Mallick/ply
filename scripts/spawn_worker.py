# Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
import modal
f = modal.Function.from_name("ply", "work")
print("spawned", f.spawn().object_id)
