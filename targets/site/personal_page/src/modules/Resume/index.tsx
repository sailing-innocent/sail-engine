import * as React from 'react';
import styles from './index.scss';
import TextBlock from '@/components/TextBlock';
import AvatarBlock from '@/components/AvatarBlock';
import AvatarSource from '@/assets/avatar_60x60.png';
import Avatar from '@/components/Avatar';
import InlineLink from '@/components/InlineLink';
import Table, { TableRow } from '@/components/Table';
import { TimeNode } from '@/components/Timeline';

const Resume = () => {
    const description = <ul>
        <li><b>NAME: </b>Zhu Zihang</li>
        <li><b>GITHUB: </b> <InlineLink href="https://github.com/sailing-innocent">https://github.com/sailing-innocent</InlineLink></li>
        <li><b>CURRENT INSTITUDE: </b>Nanjing University & D5 Inc.</li>
        <li><b>RESEARCH INTEREST: </b><br />Computer Graphics, Rendering, Modeling, Animation, AI</li>
    </ul>
    const abstract = <p>
        <b> Hello Friend! </b>
        My Name is Zhu Zihang, from China. I had a becheler degree of Control Science from Zhejiang Univesity in 2022.
        Now I am persuing a M.S degree from Nanjing University, China.
        My Education backgroud covers robotics, computer graphics and deep learning.
        My research interest is modeling and authoring, with simulation, PCG, AI or other methods. and I recently focus on NeRF, 3DGS and its variants.
        If you would like to contact me, send emails to: <br /> <InlineLink href="mailto:sailing-innocent@foxmail.com">sailing-innocent@foxmail.com</InlineLink>.
    </p>

    const expNodes: TimeNode[] = [
        {
            index: 0,
            from: new Date('1999-04-19').getTime(),
            to: null,
            node: 'birth'
        },
        {
            index: 1,
            from: new Date('2017-08-25').getTime(),
            to: new Date('2022-03-31').getTime(),
            node: "Zhejiang University CKC Cross Program on Robotics, bechelar degree on Automation"
        },
        {
            index: 2,
            from: new Date('2022-04-06').getTime(),
            to: new Date('2022-07-02').getTime(),
            node: 'Working as a FE Engineer in ByteDance Hangzhou'
        },
        {
            index: 3,
            from: new Date('2022-09-12').getTime(),
            to: null,
            node: 'Persuing M.S Degree in Nanjing University'
        },
        {
            index: 4,
            from: new Date('2022-12-21').getTime(),
            to: new Date('2023-08-21').getTime(),
            node: 'Internship in D5 focusing on AI'
        },
        {
            index: 5,
            from: new Date('2024-11-21').getTime(),
            to: new Date('2023-06').getTime(),
            node: 'Internship in D5 focusing on new Render Engine'
        },
        {
            index: 6,
            from: new Date('2024-07').getTime(),
            to: null,
            node: '(TBD'
        },
    ];

    const rows: TableRow[] = expNodes.map((expnode: TimeNode) => {
        const row: TableRow = {
            key: (expnode.index).toString(),
            ele: <div onClick={() => { }}>
                <TextBlock>
                    {expnode.node} from {new Date(expnode.from).toLocaleDateString()} to {expnode.to ? new Date(expnode.to).toLocaleDateString() : 'now'}
                </TextBlock>
            </div>
        };
        return row;
    });

    return (
        <div className={styles.container}>
            <AvatarBlock avatar={<Avatar src={AvatarSource} />} description={description} />
            <TextBlock tType='abstract'>{abstract}</TextBlock>
            <Table title={"Experience"} rows={rows}></Table>
        </div>
    )
}

export default Resume;
