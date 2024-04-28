import React from 'react';
import styles from './index.scss';
import SimpleRectangle from '@/modules/SimpleRectangle';
import Gasket2D from '@/modules/Gasket2D';
import { Tabs, TabList, TabPanels, Tab, TabPanel } from '@chakra-ui/react'

const IcgNote = () => {
    return (
        <>
            <div className={styles.container}>
                <div className={styles.main_block}>
                    <Tabs>
                        <TabList>
                            <Tab>Simple Rectangle</Tab>
                            <Tab>Gasket 2D</Tab>
                            <Tab>Three</Tab>
                        </TabList>
                        <TabPanels>
                            <TabPanel>
                                <SimpleRectangle />
                            </TabPanel>
                            <TabPanel>
                                <Gasket2D />
                            </TabPanel>
                            <TabPanel>
                                <p>three!</p>
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