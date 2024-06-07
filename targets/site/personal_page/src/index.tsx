import React from 'react';
import ReactDOM from 'react-dom/client';
import styles from './index.scss';
import Overview from '@/pages/overview';
import IcgNote from './pages/icg-note';
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ChakraProvider } from '@chakra-ui/react'

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

const container = document.getElementById('root');
const root = ReactDOM.createRoot(container);
const App = () => {
    return <ChakraProvider>
        <div className={styles.container}>
            <BrowserRouter>
                <Routes>
                    <Route path="/" element={<Overview />} />
                    <Route path="/icg-note" element={<IcgNote />} />
                </Routes>
            </BrowserRouter>
        </div>
    </ChakraProvider>

};

root.render(<App />);
