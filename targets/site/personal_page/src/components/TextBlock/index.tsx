import React from 'react';
import cx from 'classnames';
import styles from './index.scss';

export type TextBlockProps = {
    children?: React.ReactNode;
    tType?: 'passage' | 'abstract' | 'comment' | 'reference';
}

const TextBlock = (props: TextBlockProps) => {
    const {
        children = 'Sample Text',
        tType = 'passage',
    } = props;

    return (
        <div className={cx(styles.container, styles[tType])}>{children}</div>
    )
}

export default TextBlock;
