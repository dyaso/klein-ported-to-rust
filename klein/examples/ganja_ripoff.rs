// example copied from https://github.com/enkimute/ganja.js/blob/master/codegen/rust/pga3d.rs for simplicity
// (rust builds all examples when you run `cargo test`, which was showing things down.)
// please see the `demo` directory for more realistic examples

use klein::{Plane, Point, Rotor, Line, Motor, ApplyTo};

use std::f32::consts::PI;

fn main() {
	// Elements of the even subalgebra (scalar, bivectors, and the pseudoscalar) of unit length are motors
	let rot = Rotor::new(PI / 2., 0., 0., 1.);

	// The outer product ^ is the MEET. Here we intersect the yz (x=0) and xz (y=0) planes.
	let ax_z = Plane::basis_e1() ^ Plane::basis_e2();

	// line and plane meet in a point. We intersect the line along the z-axis (x=0,y=0) with the xy (z=0) plane.
	let origin = ax_z ^ Plane::basis_e3();

	// We can also easily create points and join them into a line using the regressive (vee, &) product.
	let px = Point::new(1.0, 0.0, 0.0);
	let line = origin & px;

	// Lets also create the plane with equation 2x + z - 3 = 0
    // A plane is defined using its homogenous equation ax + by + cz + d = 0
	let p = Plane::new(2.0, 0.0, 1.0, -3.0);

	// rotations work on all elements
	let rotated_plane = rot.apply_to(p);
	let rotated_line  = rot.apply_to(line);
	let rotated_point = rot.apply_to(px);

	// See the 3D PGA Cheat sheet for a huge collection of useful formulas
	let point_on_plane = (p | px) ^ p;

	// Some output
	println!("a point       : {}", px);
	println!("a line        : {}", line);
	println!("a plane       : {}", p);
	println!("a rotor       : {}", rot);
	println!("rotated line  : {}", rotated_line);
	println!("rotated point : {}", rotated_point);
	println!("rotated plane : {}", rotated_plane);
	println!("point on plane: {}", point_on_plane.normalized());
	// println!("point on torus: {}", PGA3D::point_on_torus(0.0, 0.0));
	// println!("{}", PGA3D::e0()-1.0);
	// println!("{}", 1.0-PGA3D::e0());
}