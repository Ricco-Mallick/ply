import { createRoot } from 'react-dom/client';
import '../shared/tokens.css';
import '../walk/walk.css';
import './lab.css';
import { Lab } from './Lab';

createRoot(document.getElementById('root')!).render(<Lab />);
