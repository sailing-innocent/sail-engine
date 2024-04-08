import React from 'react';
import styles from './index.scss';

export type TableRow = {
    key: string;
    ele: React.ReactNode;
}

export type TableProps = {
    rows?: TableRow[]
    title?: string;
}

const Table = (props: TableProps) => {
    const {
        rows = [],
        title = "Sample Table"
    } = props;
    return (
        <div className={styles.container}>
            <div className={styles.title}>{title}</div>
            <ul>
                {rows.map((row: TableRow) => {
                    return <li key={row.key}>{row.ele}</li>
                })}
            </ul>
        </div>

    )
}

export default Table;
