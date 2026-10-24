// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
import { createRoot } from 'react-dom/client';
import '../shared/tokens.css';
import '../walk/walk.css';
import '../shared/places.css';
import './lab.css';
import { Lab } from './Lab';

createRoot(document.getElementById('root')!).render(<Lab />);
