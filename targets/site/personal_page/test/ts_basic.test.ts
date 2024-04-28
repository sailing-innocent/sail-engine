const user = {
    name: "John",
    id: 0,
};

interface User {
    name: string;
    id: number;
}

const interfaced_user: User = {
    name: "John",
    id: 0,
};

interface Point {
    x: number;
    y: number;
};


test("obj_test", ()=>{
    expect(user.name).toBe("John");
    expect(user.id).toBe(0);
})

interface DummyInterface {
    return_num(): number;
}

class DummyClass implements DummyInterface {
    return_num(): number {
        return 1;
    }
}

class DummyClass2 implements DummyInterface {
    return_num(): number {
        return 2;
    }
}
test("interface_test", ()=>{
    expect(interfaced_user.name).toBe("John");
    expect(interfaced_user.id).toBe(0);
    const point: Point = { x: 12, y: 26 };
    expect(point.x).toBe(12);
    expect(point.y).toBe(26);
    const dummy1 = new DummyClass();
    const dummy2 = new DummyClass2();
    expect(dummy1.return_num()).toBe(1);
    expect(dummy2.return_num()).toBe(2);
});


// class extends
// interface extends
// type PointInstance = InstanceType<typeof Point>;
// abstract class
