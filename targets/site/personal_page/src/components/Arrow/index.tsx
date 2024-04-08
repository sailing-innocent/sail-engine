import React from 'react';

export type ArrowProps = {
    length?: number;
    width?: number;
    color?: string;
    dir?: 'horizontal' | 'vertical';
}

const Arrow = (props: ArrowProps) => {
    const { length = 100, width = 30, dir = 'horizontal', color = '#000' } = props;
    switch (dir) {
        case 'vertical':
            return (
                <svg width={width.toString()} height={length.toString()} version="1.1" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <marker id="arrow" markerUnits="strokeWidth" markerWidth="6" markerHeight="6" viewBox="0 0 6 6" refX="6"
                            refY="3" orient="auto">
                            <path d="M1,1 L5,3 L1,5 L1,1" style={{ fill: color }} />
                        </marker>
                    </defs>
                    <line x1={(width / 2).toString()} y1={'0'} x2={(width / 2).toString()} y2={length.toString()} stroke={color} strokeWidth='2'
                        markerMid='url(#arrow)' markerEnd='url(#arrow)'>
                    </line>
                </svg>
            )
        case 'horizontal':
            return (
                <svg width={length} height={width} version="1.1" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <marker id="arrow" markerUnits="strokeWidth" markerWidth="6" markerHeight="6" viewBox="0 0 6 6" refX="6"
                            refY="3" orient="auto">
                            <path d="M1,1 L5,3 L1,5 L1,1" style={{ fill: color }} />
                        </marker>
                    </defs>
                    <line y1={(width / 2).toString()} x1={'0'} y2={(width / 2).toString()} x2={length.toString()} stroke={color} strokeWidth='2'
                        markerMid='url(#arrow)' markerEnd='url(#arrow)'>
                    </line>
                </svg>
            )
        default:
            return null;
    }
}

export default Arrow;