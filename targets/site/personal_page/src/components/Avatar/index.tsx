import * as React from 'react'
// wrap a picture to avatar node
import DefaultAvatar from '@/assets/avatar_60x60.png';
import styles from './index.scss';

export type AvatarProps = {
    src?: string;
    size?: 'large' | 'normal' | 'small';
}

const Avatar = (props: AvatarProps) => {
    const {
        src = DefaultAvatar,
        size = 'normal'
    } = props;

    return (
        <div className={styles.container}>
            <img className={styles.img} src={src} alt="avatar"></img>
        </div>
    )
}

export default Avatar;
