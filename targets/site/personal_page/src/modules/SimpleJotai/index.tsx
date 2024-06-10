import { atom, useAtom } from 'jotai';

const textAtom = atom('Hello, Jotai!');
const uppercaseAtom = atom((get) => get(textAtom).toUpperCase());

const Input = () => {
    const [text, setText] = useAtom(textAtom);
    return (
        <input value={text} onChange={(e) => setText(e.target.value)} />
    );
}

const Uppercase = () => {
    const uppercase = useAtom(uppercaseAtom);
    return <div>{uppercase}</div>;
}

export { Input, Uppercase };