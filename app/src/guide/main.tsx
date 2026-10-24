// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
import { createRoot } from 'react-dom/client';
import '../shared/tokens.css';
import './guide.css';
import { Guide } from './Guide';

createRoot(document.getElementById('root')!).render(<Guide />);
