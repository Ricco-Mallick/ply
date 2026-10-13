import { createRoot } from 'react-dom/client';
import '../shared/tokens.css';
import './walk.css';
import '../shared/places.css';
import { Walk } from './Walk';

createRoot(document.getElementById('root')!).render(<Walk />);
