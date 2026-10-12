import { createRoot } from 'react-dom/client';
import '../shared/tokens.css';
import './guide.css';
import { Guide } from './Guide';

createRoot(document.getElementById('root')!).render(<Guide />);
