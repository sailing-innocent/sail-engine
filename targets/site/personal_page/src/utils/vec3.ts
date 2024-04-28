
export class vec3 {
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

export function mix(a: vec3, b: vec3, t: number) {
    return a.mulNumber(1.0 - t).add(b.mulNumber(t));
}