
test("vec3_test", ()=>{
    class vec3 {
        x: number;
        y: number;
        z: number;
        constructor(x: number, y: number, z: number) {
            this.x = x;
            this.y = y;
            this.z = z;
        }
        // +-*/%+=- operators
        add(v: vec3) {
            return new vec3(this.x + v.x, this.y + v.y, this.z + v.z);
        }
        sub(v: vec3) {
            return new vec3(this.x - v.x, this.y - v.y, this.z - v.z);
        }
        mul(v: vec3) {
            return new vec3(this.x * v.x, this.y * v.y, this.z * v.z);
        }
        div(v: vec3) {
            return new vec3(this.x / v.x, this.y / v.y, this.z / v.z);
        }
        // number
        addNumber(n: number) {
            return new vec3(this.x + n, this.y + n, this.z + n);
        }
        subNumber(n: number) {
            return new vec3(this.x - n, this.y - n, this.z - n);
        }
        mulNumber(n: number) {
            return new vec3(this.x * n, this.y * n, this.z * n);
        }
        divNumber(n: number) {
            return new vec3(this.x / n, this.y / n, this.z / n);
        }
    }
    const points: vec3[] = [];
    function triangle(a: vec3, b: vec3, c: vec3) {
        points.push(a);
        points.push(b);
        points.push(c);
    };
    
    const a = new vec3(1, 2, 3);
    const b = new vec3(4, 5, 6);
    const c = new vec3(7, 8, 9);
    triangle(a, b, c);
    expect(points[0].x).toBe(1);
    expect(points[0].y).toBe(2);
    expect(points[0].z).toBe(3);
    expect(points[1].x).toBe(4);
    expect(points[1].y).toBe(5);
    expect(points[1].z).toBe(6);
    expect(points[2].x).toBe(7);
    expect(points[2].y).toBe(8);
    expect(points[2].z).toBe(9);
});