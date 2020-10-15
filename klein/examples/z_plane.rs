use druid::kurbo::{BezPath,Circle};
use druid::piet::{FontFamily, ImageFormat, InterpolationMode, FontWeight, FontStyle,

};

use druid::kurbo;

use druid::widget::prelude::*;
use druid::{
    Affine, AppLauncher, ArcStr, Color, FontDescriptor, LocalizedString, Point as DruidPoint, Rect, TextLayout,
    WindowDesc,
    Data, MouseEvent
};

use klein::{Plane, Point, Line
	};

#[derive(Default)]
struct CustomWidget {
	lower_y: f64,
	upper_y: f64,
	 left_x: f64,
	right_x: f64,

    lower_plane: Plane,
    upper_plane: Plane,
     left_plane: Plane,
    right_plane: Plane,

    //center_point: Rc<PGA3D>,
    center_point: Point,
    left: f64, top: f64, scale: f64,

    window_pixels: kurbo::Size,
}

impl CustomWidget {
	fn new() -> CustomWidget {
		CustomWidget {
			lower_y: -2.,
			upper_y:  2.,
			left_x:  -2.,
			right_x:  2.,
			..Default::default()}
	}
	pub fn set_window_boundary_planes(&mut self, state: &State, window: &kurbo::Size) {
		println!("HELSES");
        let desired_width  = self.right_x - self.left_x;
        let desired_height = self.upper_y - self.lower_y;
        let desired_aspect_ratio = desired_width / desired_height;

        let center_x = (self.right_x + self.left_x)  / 2.;
        let center_y = (self.upper_y + self.lower_y) / 2.;

        self.window_pixels = *window;
        let window_aspect_ratio = window.width / window.height;

        self.top       = self.upper_y;
        self.left      = self.left_x;
        let mut right  = self.right_x;
        let mut bottom = self.lower_y;

        if window_aspect_ratio > desired_aspect_ratio {
            // actual window is wider than desired viewport
            self.scale = desired_height / window.height;
            let half_width = self.scale * 0.5 * window.width;
            right = center_x + half_width;
            self.left  = center_x - half_width;
        } else {
            // actual window is taller than desired viewport
            self.scale = desired_width / window.width;
            let half_height = self.scale * 0.5 * window.height;
            self.top    = center_y + half_height;
            bottom = center_y - half_height;
        }

        self.lower_plane  = Plane::new(0., 1., 0., bottom as f32);
        self.upper_plane  = Plane::new(0., 1., 0., self.top  as f32);
        self.left_plane   = Plane::new(1.,  0., 0., self.left as f32);
        self.right_plane  = Plane::new(1.,  0., 0., right  as f32);

        self.center_point = Point::new(center_x as f32, center_y as f32, 0.);
    }

    fn dist(a :&Point, b :&Point) -> f32 {
        let d = (a.normalized() & b.normalized()).norm();
println!("dist: {}",d);

        return if f32::is_nan(d) {1000000.} else {d}
    }

    pub fn draw_line(&self, ctx: &mut PaintCtx, line: &Line) {
        let mut intersections = Vec::<Point>::new();
        intersections.push(*line ^ self.lower_plane);
        intersections.push(*line ^ self.upper_plane);
        intersections.push(*line ^ self.left_plane);
        intersections.push(*line ^ self.right_plane);

        intersections.sort_by(|a,b| CustomWidget::dist(a, &self.center_point)
                                        .partial_cmp(
                                            &CustomWidget::dist(b, &self.center_point)).unwrap());

        let end1 = &intersections[0].normalized();
        let end2 = &intersections[1].normalized();
self.draw_point(ctx,end1);
self.draw_point(ctx,end2);
        let mut path = BezPath::new();
        path.move_to(DruidPoint::new(
            ((end1.x() as f64 - self.left) / self.scale),
            ((self.top - end1.y()  as f64) / self.scale)));
        path.line_to(DruidPoint::new(
            ((end2.x() as f64 - self.left) / self.scale), 
            ((self.top - end2.y() as f64)  / self.scale)));
        let stroke_color = Color::rgb8(150,150,150);
        ctx.stroke(path, &stroke_color, 1.0);

    }

    pub fn draw_point(&self, ctx: &mut PaintCtx, point: &Point) {
        let fill_color = Color::rgba8(0xa3, 0xff, 0xff, 0xFF);

        ctx.fill(
            Circle::new(DruidPoint::new(
                ((point.x() as f64 - self.left) / self.scale), 
                ((self.top - point.y() as f64)  / self.scale)), 15.0),
                // ((1.0 - self.left) / self.scale), 
                // ((self.top - 1.0)  / self.scale)), 15.0),
            &fill_color,
            
        );

    }

    pub fn mouse_move(&self, mouse: &MouseEvent) {
        let x_portion = (mouse.pos.x / self.window_pixels.width);
        let x_coord = self.left_plane.x()*self.left_plane.d()*(1.-x_portion as f32) + (self.right_plane.x()*self.right_plane.d())*x_portion as f32;
        let y_portion = (mouse.pos.y / self.window_pixels.height);
        let y_coord = self.upper_plane.y()*self.upper_plane.d()*(1.-y_portion as f32) + self.lower_plane.y()*self.lower_plane.d()*y_portion as f32;
        // let mouse_point = (x_plane^y_plane.normalized()^PGA3D::e3());
        // let mouse_y = mouse.pos.y;
        // let y_plane = 
        println!("plane {} {}",self.upper_plane.y(),self.upper_plane.d());
        println!("mosue pos {} {}",x_coord,y_coord);
        // println!("mosue pos {} {}",mouse_point.get032(), mouse_point.get013())
    }

}

#[derive(Clone,Data,Default)]
struct State {
}

impl Widget<State> for CustomWidget {
    fn event(&mut self, ctx: &mut EventCtx, event: &Event, _data: &mut State, _env: &Env) {
        match event {
            Event::WindowConnected => {
                ctx.request_focus();
            }
            Event::KeyDown(e) => {
                println!("key down event {:?}", e);
            }
            Event::MouseMove(e) => {
                self.mouse_move(e);
            }
            _ => {
                println!("unhandled input event {:?}", event);
            }
        }
    }

    fn lifecycle(
        &mut self,
        ctx: &mut LifeCycleCtx,
        event: &LifeCycle,
        data: &State,
        _env: &Env,
    ) {
        match event {
            LifeCycle::Size(s) => {
                self.set_window_boundary_planes(data, s);
            }
            LifeCycle::WidgetAdded => {
               ctx.register_for_focus();
            }
            _ => {println!("unhandled lifecycle event: {:?}", event)
            }
        }
    }

    fn update(&mut self, _ctx: &mut UpdateCtx, _old_data: &State, _data: &State, _env: &Env) {
        println!("update event: {}",0);
    }

    fn layout(
        &mut self,
        _layout_ctx: &mut LayoutCtx,
        bc: &BoxConstraints,
        _data: &State,
        _env: &Env,
    ) -> Size {
        // BoxConstraints are passed by the parent widget.
        // This method can return any Size within those constraints:
        // bc.constrain(my_size)
        //
        // To check if a dimension is infinite or not (e.g. scrolling):
        // bc.is_width_bounded() / bc.is_height_bounded()
        bc.max()
    }


    // The paint method gets called last, after an event flow.
    // It goes event -> update -> layout -> paint, and each method can influence the next.
    // Basically, anything that changes the appearance of a widget causes a paint.
    fn paint(&mut self, ctx: &mut PaintCtx
        , data: &State, env: &Env) {
        // Let's draw a picture with Piet!

        // Clear the whole widget with the color of your choice
        // (ctx.size() returns the size of the layout rect we're painting in)
        let size = ctx.size();
        let rect = size.to_rect();
        ctx.fill(rect, &Color::BLACK);

        // // for p in (&data.points).borrow().iter() {
        // for p in (&data.points).iter() {
        //     println!("label: {}", p.label);
        // }

        // Note: ctx also has a `clear` method, but that clears the whole context,
        // and we only want to clear this widget's area.

        // Create an arbitrary bezier path
        // let mut path = BezPath::new();
        // path.move_to(Point::ORIGIN);
        // path.quad_to((80.0, 90.0), (size.width, size.height));
        // // Create a color
        // let stroke_color = Color::rgb8(255,0, 255);
        // // Stroke the path with thickness 1.0
        // ctx.stroke(path, &stroke_color, 1.0);


        let p1 = Point::new(0.,   1., 0.);
        let p2 = Point::new(0.94, 0.,  0.);

         let l = &(p1&p2);

        self.draw_point(ctx, &p1);
        self.draw_point(ctx, &p2);

         self.draw_line(ctx, l);

//         // 0 = same plane (intersect at origin)
//         // -1e02 parallel planes separated but parallel
//         // 0.0995037e01 + -0.9950371e02 + -0.0995037e12 intersect in line

//         // Rectangles: the path for practical people
// //        let rect = Rect::from_origin_size((10., 10.), (100., 100.));
//         // Note the Color:rgba8 which includes an alpha channel (7F in this case)
	    let fill_color = Color::rgba8(0xa3, 0xa3, 0xa3, 0xFF);
// //        ctx.fill(rect, &fill_color);


//         ctx.stroke(
//             Circle::new(Point::new(data.x as f64, data.y as f64), 15.0),
//             &fill_color,
//             5.0,
//         );



        // Text is easy; in real use TextLayout should be stored in the widget
        // and reused.
        let mut layout = TextLayout::<ArcStr>::from_text("hello");//data.to_owned());
        layout.set_font(FontDescriptor::new(FontFamily::SANS_SERIF).
            with_size(24.0)//.with_weight(FontWeight::BOLD)
            .with_style(FontStyle::Italic));
        layout.set_text_color(fill_color);
//        layout.set_text_style(FontStyle::Italic);
        layout.rebuild_if_needed(ctx.text(), env);

        // Let's rotate our text slightly. First we save our current (default) context:
        ctx.with_save(|ctx| {
            // Now we can rotate the context (or set a clip path, for instance):
            ctx.transform(Affine::rotate(0.0));
            layout.draw(ctx, (80.0, 40.0));
        });
        // When we exit with_save, the original context's rotation is restored

//drawtext(ctx);
     // let layout2 = ctx
        // .text()

// let mut moo: () = layout2;
        // .new_text_layout("Helloo piet!");
        // .font(FontFamily::SYSTEM_UI, 24.0)
// .default_attribute(FontStyle::Italic)
// .default_attribute(FontWeight::BOLD)
        // .default_attribute(TextAttribute::TextColor(RED_ALPHA))
        // .build()?;

//     let w: f64 = layout2.size().width;
//     rc.draw_text(&layout2, (80.0, 10.0));

//     rc.stroke(Line::new((80.0, 12.0), (80.0 + w, 12.0)), &RED_ALPHA, 1.0);

//     rc.with_save(|rc| {
//         rc.transform(Affine::rotate(0.1));
//         rc.draw_text(&layout2, (80.0, 10.0));
//         Ok(())
//     })?;



        // Let's burn some CPU to make a (partially transparent) image buffer
        let image_data = make_image_data(256, 256);
        let image = ctx
            .make_image(256, 256, &image_data, ImageFormat::RgbaSeparate)
            .unwrap();
        // The image is automatically scaled to fit the rect you pass to draw_image
        ctx.draw_image(&image, size.to_rect(), InterpolationMode::Bilinear);
    }
}






pub fn main() {
    let window = WindowDesc::new(|| CustomWidget::new()).title(
        LocalizedString::new("custom-widget-demo-window-title").with_placeholder("z=0 plane 2d graphics"),
    );

    let mut s = State{..Default::default()};

    AppLauncher::with_window(window)
        .use_simple_logger()
        .launch(s)
        // .launch("Druid + Piet".to_string())
        .expect("launch failed");
}



fn make_image_data(width: usize, height: usize) -> Vec<u8> {
    let mut result = vec![0; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let ix = (y * width + x) * 4;
            result[ix] = x as u8;
            result[ix + 1] = y as u8;
            result[ix + 2] = !(x as u8);
            result[ix + 3] = 127;
        }
    }
    result
}
