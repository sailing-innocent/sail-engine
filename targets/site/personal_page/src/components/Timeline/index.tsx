import React from 'react';
import cx from 'classnames';
import styles from './index.scss';
import Arrow from '../Arrow';
import TextBlock from '../TextBlock';

export type TimeNode = {
    from?: number;
    to?: number;
    node?: React.ReactNode;
    index?: number;
    offset?: number;
    width?: number;
}

export type TimeLineProps = {
    nodes?: TimeNode[];
    title?: string;
    dir?: 'vertical' | 'horizontal';
    length?: number;
    maxItem?: number;
}

const fontSize = 12;

const TimeLine = (props: TimeLineProps) => {
    return <div className={styles.container}>Timeline</div>
}

export default TimeLine;
