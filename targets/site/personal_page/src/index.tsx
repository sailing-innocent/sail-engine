import * as React from 'react';
import * as ReactDOM from 'react-dom/client';
import styles from './index.scss';
import Overview from '@/pages/overview';

// prevent control zoom in/out
document.addEventListener('keydown', function (event) {
    if (event.ctrlKey === true || event.metaKey === true) {
        event.preventDefault();
    }
}, false);

// prevent up/down scroll
document.body.addEventListener("wheel", (e) => {
    if (e.ctrlKey) {
        if (e.deltaY < 0) {
            e.preventDefault();
            return false;
        }
        if (e.deltaY > 0) {
            e.preventDefault();
            return false;
        }
    }
}, { passive: false });
const root = ReactDOM.createRoot(document.body);
const App = () => {
    return <div className={styles.container}>
        <Overview />
    </div>
};

root.render(<App />);
