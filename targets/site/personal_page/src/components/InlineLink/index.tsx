import React from 'react';
import styles from './index.scss';

export type InlineLinkProps = {
    href: string;
    children?: React.ReactNode;
}

const InlineLink = (props: InlineLinkProps) => {
    const {
        href,
        children,
    } = props;
    return <a href={href} className={styles.inlineLink}>{children ?? href}</a>
}

export default InlineLink;
