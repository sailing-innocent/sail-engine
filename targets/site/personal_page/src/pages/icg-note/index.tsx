import React from 'react';
import styles from './index.scss';
import SimpleTriangle from '@/modules/SimpleTriangle';
import DevideTriangle from '@/modules/DevideTriangle';
import DevideTetra from '@/modules/DevideTetra';
import { Tabs, TabList, TabPanels, Tab, TabPanel } from '@chakra-ui/react'

const IcgNote = () => {
    return (
        <>
            <div className={styles.container}>
                <div className={styles.main_block}>
                    <Tabs>
                        <TabList>
                            <Tab>Simple Triangle</Tab>
                            <Tab>Devide Triangle</Tab>
                            <Tab>Device Tetra</Tab>
                        </TabList>
                        <TabPanels>
                            <TabPanel>
                                <SimpleTriangle />
                            </TabPanel>
                            <TabPanel>
                                <DevideTriangle />
                            </TabPanel>
                            <TabPanel>
                                <DevideTetra />
                            </TabPanel>
                        </TabPanels>
                    </Tabs>

                </div>
                <div className={styles.side_block}>
                </div>
            </div>
        </>
    )
}

export default IcgNote;