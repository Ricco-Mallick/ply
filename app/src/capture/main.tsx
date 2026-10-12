import { createRoot } from 'react-dom/client';
import '../shared/tokens.css';
import './capture.css';
import { Capture } from './Capture';

createRoot(document.getElementById('root')!).render(<Capture />);
