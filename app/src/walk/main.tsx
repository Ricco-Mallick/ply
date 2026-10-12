import { createRoot } from 'react-dom/client';
import '../shared/tokens.css';
import './walk.css';
import { Walk } from './Walk';

createRoot(document.getElementById('root')!).render(<Walk />);
