import {useEffect, useState} from 'react';

interface CanvasSize {
    width: number;
    height: number;
};

const asideWidth = 500;
const asideHeight = 300;

const useCanvasSize = () => {
    const [canvasSize, setCanvasSize] = useState<CanvasSize>(
        { width: 1, height: 1});
    useEffect(()=>{
        const handleResize = () => {
            setCanvasSize({
                width: document.body.clientWidth - asideWidth,
                height: document.body.clientHeight
            });
        }
        handleResize();
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
        }
    }, []);

    return canvasSize;
}

export default useCanvasSize;