import React from 'react';
import TextBlock from '../TextBlock';
import styles from './index.scss';

export type AvatarBlockProps = {
    avatar: React.ReactNode;
    description: React.ReactNode;
}

const AvatarBlock = (props: AvatarBlockProps) => {
    const {
        avatar,
        description,
    } = props;

    return (
        <div className={styles.container}>
            <div className={styles.avatar}>{avatar}</div>
            <div className={styles.description}>
                <TextBlock>{description}</TextBlock>
            </div>
        </div>
    )
}

export default AvatarBlock;
