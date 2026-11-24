// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* The twin a page opens when the URL names none: the team's own chapel capture. Round 2 allows own captures only,
   so no third-party scene is ever a default. */
const API = ((import.meta.env.VITE_API as string | undefined) || 'https://ply-api.ricco-ply.workers.dev').replace(/\/$/, '');

export const DEFAULT_SCENE = `${API}/v1/scenes/chapel-p7-1/scene.json`;
