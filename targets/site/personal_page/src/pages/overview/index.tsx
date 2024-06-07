import * as React from 'react';
import styles from './index.scss';
import SimpleTriangle from '@/modules/SimpleTriangle';
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
                    <SimpleTriangle />
                </div>
            </div>
        </>
    )
}

export default Overview;