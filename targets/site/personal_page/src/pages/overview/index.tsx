import * as React from 'react';
import styles from './index.scss';
import SimpleRectangle from '@/modules/SimpleRectangle';
import Resume from '@/modules/Resume';
import { Link } from 'react-router-dom';

const Overview = () => {
    return (
        <>
            <div className={styles.container}>
                <div className={styles.side_block}>
                    <Resume />
                    <Link to="/icg-note">ICG Note</Link>
                </div>
                <div className={styles.main_block}>
                    <SimpleRectangle />
                </div>
            </div>
        </>
    )
}

export default Overview;