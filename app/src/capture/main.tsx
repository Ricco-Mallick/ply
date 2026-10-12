import { createRoot } from 'react-dom/client';
import '../shared/tokens.css';
import './capture.css';
import { App } from './App';

createRoot(document.getElementById('root')!).render(<App />);
