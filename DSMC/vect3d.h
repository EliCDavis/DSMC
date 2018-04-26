#ifndef VECT3D_H
#define VECT3D_H
#include <cmath>
#include <iostream>

//---------------------vect3d------------------//
struct vect3d {
	float x, y, z;
	vect3d() {}
	vect3d(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
	vect3d(const vect3d &v) { x = v.x; y = v.y; z = v.z; }
};

inline std::ostream & operator<<(std::ostream &s, const vect3d &v) {
	s << v.x << ' ' << v.y << ' ' << v.z << ' ';
	return s;
}

inline std::istream &operator>>(std::istream &s, vect3d &v) {
	s >> v.x >> v.y >> v.z;
	return s;
}

inline float dot(const vect3d &v1, const vect3d &v2) {
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

inline float norm(const vect3d &v) {
	return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

inline vect3d cross(const vect3d &v1, const vect3d &v2) {
	return vect3d(v1.y*v2.z - v1.z*v2.y,
		v1.z*v2.x - v1.x*v2.z,
		v1.x*v2.y - v1.y*v2.x);
}

inline vect3d &operator*=(vect3d &target, float val) {
	target.x *= val;
	target.y *= val;
	target.z *= val;
	return target;
}

inline vect3d &operator/=(vect3d &target, float val) {
	target.x /= val;
	target.y /= val;
	target.z /= val;
	return target;
}

inline vect3d &operator*=(vect3d &target, double val) {
	target.x *= val;
	target.y *= val;
	target.z *= val;
	return target;
}

inline vect3d &operator/=(vect3d &target, double val) {
	target.x /= val;
	target.y /= val;
	target.z /= val;
	return target;
}

inline vect3d &operator*=(vect3d &target, long double val) {
	target.x *= val;
	target.y *= val;
	target.z *= val;
	return target;
}

inline vect3d &operator/=(vect3d &target, long double val) {
	target.x /= val;
	target.y /= val;
	target.z /= val;
	return target;
}

inline vect3d operator+=(vect3d &target, const vect3d &val) {
	target.x += val.x;
	target.y += val.y;
	target.z += val.z;
	return target;
}

inline vect3d operator-=(vect3d &target, const vect3d &val) {
	target.x -= val.x;
	target.y -= val.y;
	target.z -= val.z;
	return target;
}

inline vect3d operator+(const vect3d &v1, const vect3d &v2) {
	return vect3d(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

vect3d operator-(const vect3d &v1, const vect3d &v2) {
	return vect3d(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

inline vect3d operator*(const vect3d &v1, float r2) {
	return vect3d(v1.x*r2, v1.y*r2, v1.z*r2);
}

inline vect3d operator*(float r1, const vect3d &v2) {
	return vect3d(v2.x*r1, v2.y*r1, v2.z*r1);
}

inline vect3d operator/(const vect3d &v1, float r2) {
	return vect3d(v1.x / r2, v1.y / r2, v1.z / r2);
}

inline vect3d operator*(const vect3d &v1, double r2) {
	return vect3d(v1.x*r2, v1.y*r2, v1.z*r2);
}

inline vect3d operator*(double r1, const vect3d &v2) {
	return vect3d(v2.x*r1, v2.y*r1, v2.z*r1);
}

inline vect3d operator/(const vect3d &v1, double r2) {
	return vect3d(v1.x / r2, v1.y / r2, v1.z / r2);
}

inline vect3d operator*(const vect3d &v1, long double r2) {
	return vect3d(v1.x*r2, v1.y*r2, v1.z*r2);
}

inline vect3d operator*(long double r1, const vect3d &v2) {
	return vect3d(v2.x*r1, v2.y*r1, v2.z*r1);
}

inline vect3d operator/(const vect3d &v1, long double r2) {
	return vect3d(v1.x / r2, v1.y / r2, v1.z / r2);
}
#endif
