import * as React from 'react';
import styles from './index.scss';
import SimpleRectangle from '@/modules/SimpleRectangle';
import Resume from '@/modules/Resume';

const Overview = () => {
    return (
        <>
            <div className={styles.container}>
                <div className={styles.side_block}>
                    <Resume />
                </div>
                <div className={styles.main_block}>
                    <SimpleRectangle />
                </div>

            </div>
        </>
    )
}

export default Overview;