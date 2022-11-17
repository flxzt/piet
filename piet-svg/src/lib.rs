//! SVG output support for piet

#![deny(clippy::trivially_copy_pass_by_ref)]

#[cfg(feature = "evcxr")]
mod evcxr;
mod text;

use std::{borrow::Cow, fmt, fmt::Write, io, mem};

use image::{DynamicImage, GenericImageView, ImageBuffer};
use piet::kurbo::{Affine, BezPath, PathEl, Point, Rect, Shape, Size};
use piet::{
    Color, Error, FixedGradient, FontStyle, Image, ImageFormat, InterpolationMode, IntoBrush,
    LineCap, LineJoin, StrokeStyle, TextAlignment, TextLayout as _,
};
use svg::node::Node;

pub use crate::text::{Text, TextLayout};
// re-export piet
pub use piet;

#[cfg(feature = "evcxr")]
pub use evcxr::draw_evcxr;

type Result<T> = std::result::Result<T, Error>;

/// `piet::RenderContext` for generating SVG images
pub struct RenderContext {
    bounds: Rect,
    viewbox: Rect,
    decimal_digits: usize,
    stack: Vec<State>,
    state: State,
    doc: svg::Document,
    next_id: u64,
    text: Text,
}

impl RenderContext {
    /// Construct an empty `RenderContext`
    pub fn new(bounds: Rect, viewbox: Rect, decimal_digits: usize) -> Self {
        Self {
            bounds,
            viewbox,
            decimal_digits,
            stack: Vec::new(),
            state: State::default(),
            doc: svg::Document::new(),
            next_id: 0,
            text: Text::new(),
        }
    }

    /// Construct a `RenderContext` without text sources
    pub fn new_no_text(bounds: Rect, viewbox: Rect, decimal_digits: usize) -> Self {
        Self {
            bounds,
            viewbox,
            decimal_digits,
            stack: Vec::new(),
            state: State::default(),
            doc: svg::Document::new(),
            next_id: 0,
            text: Text::new_empty_source(),
        }
    }

    // An escape hatch to directly append svg nodes to the context
    pub fn append_svg_node(&mut self, node: impl Node) {
        self.doc.append(node);
    }

    /// The size that the SVG will render at.
    pub fn size(&self) -> Size {
        self.viewbox.size()
    }

    /// Write graphics rendered so far to an `std::io::Write` impl, such as `std::fs::File`
    ///
    /// Additional rendering can be done afterwards.
    pub fn write(&self, writer: impl io::Write) -> io::Result<()> {
        svg::write(writer, &self.doc)
    }

    /// Returns an object that can write the svg somewhere.
    pub fn display(&self) -> &impl fmt::Display {
        &self.doc
    }

    fn new_id(&mut self) -> Id {
        let x = Id(self.next_id);
        self.next_id += 1;
        x
    }
}

impl piet::RenderContext for RenderContext {
    type Brush = Brush;

    type Text = Text;
    type TextLayout = TextLayout;

    type Image = SvgImage;

    fn status(&mut self) -> Result<()> {
        Ok(())
    }

    fn clear(&mut self, rect: impl Into<Option<Rect>>, color: Color) {
        let rect = rect.into();
        let mut rect = match rect {
            Some(rect) => svg::node::element::Rectangle::new()
                .set("width", format!("{:.*}", self.decimal_digits, rect.width()))
                .set(
                    "height",
                    format!("{:.*}", self.decimal_digits, rect.height()),
                )
                .set("x", format!("{:.*}", self.decimal_digits, rect.x0))
                .set("y", format!("{:.*}", self.decimal_digits, rect.y0)),
            None => svg::node::element::Rectangle::new()
                .set("width", "100%")
                .set("height", "100%"),
        }
        .set("fill", fmt_color(color))
        .set("fill-opacity", fmt_opacity(color));
        //FIXME: I don't think we should be clipping, here?
        if let Some(id) = self.state.clip {
            rect.assign("clip-path", format!("url(#{})", id.to_string()));
        }
        self.doc.append(rect);
    }

    fn solid_brush(&mut self, color: Color) -> Brush {
        Brush {
            kind: BrushKind::Solid(color),
        }
    }

    fn gradient(&mut self, gradient: impl Into<FixedGradient>) -> Result<Brush> {
        let id = self.new_id();
        match gradient.into() {
            FixedGradient::Linear(x) => {
                let mut gradient = svg::node::element::LinearGradient::new()
                    .set("gradientUnits", "userSpaceOnUse")
                    .set("id", id)
                    .set("x1", format!("{:.*}", self.decimal_digits, x.start.x))
                    .set("y1", format!("{:.*}", self.decimal_digits, x.start.y))
                    .set("x2", format!("{:.*}", self.decimal_digits, x.end.x))
                    .set("y2", format!("{:.*}", self.decimal_digits, x.end.y));
                for stop in x.stops {
                    gradient.append(
                        svg::node::element::Stop::new()
                            .set("offset", format!("{:.*}", self.decimal_digits, stop.pos))
                            .set("stop-color", fmt_color(stop.color))
                            .set("stop-opacity", fmt_opacity(stop.color)),
                    );
                }
                self.doc.append(gradient);
            }
            FixedGradient::Radial(x) => {
                let mut gradient = svg::node::element::RadialGradient::new()
                    .set("gradientUnits", "userSpaceOnUse")
                    .set("id", id)
                    .set("cx", format!("{:.*}", self.decimal_digits, x.center.x))
                    .set("cy", format!("{:.*}", self.decimal_digits, x.center.y))
                    .set(
                        "fx",
                        format!("{:.*}", self.decimal_digits, x.center.x + x.origin_offset.x),
                    )
                    .set(
                        "fy",
                        format!("{:.*}", self.decimal_digits, x.center.y + x.origin_offset.y),
                    )
                    .set("r", format!("{:.*}", self.decimal_digits, x.radius));
                for stop in x.stops {
                    gradient.append(
                        svg::node::element::Stop::new()
                            .set("offset", format!("{:.*}", self.decimal_digits, stop.pos))
                            .set("stop-color", fmt_color(stop.color))
                            .set("stop-opacity", fmt_opacity(stop.color)),
                    );
                }
                self.doc.append(gradient);
            }
        }
        Ok(Brush {
            kind: BrushKind::Ref(id),
        })
    }

    fn fill(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        add_shape(
            &mut self.doc,
            shape,
            &Attrs {
                xf: self.state.xf,
                clip: self.state.clip,
                fill: Some((brush.into_owned(), None)),
                ..Attrs::default()
            },
            self.decimal_digits,
        );
    }

    fn fill_even_odd(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        add_shape(
            &mut self.doc,
            shape,
            &Attrs {
                xf: self.state.xf,
                clip: self.state.clip,
                fill: Some((brush.into_owned(), Some("evenodd"))),
                ..Attrs::default()
            },
            self.decimal_digits,
        );
    }

    fn clip(&mut self, shape: impl Shape) {
        let id = self.new_id();
        let mut clip = svg::node::element::ClipPath::new().set("id", id);
        add_shape(
            &mut clip,
            shape,
            &Attrs {
                xf: self.state.xf,
                clip: self.state.clip,
                ..Attrs::default()
            },
            self.decimal_digits,
        );
        self.doc.append(clip);
        self.state.clip = Some(id);
    }

    fn stroke(&mut self, shape: impl Shape, brush: &impl IntoBrush<Self>, width: f64) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        add_shape(
            &mut self.doc,
            shape,
            &Attrs {
                xf: self.state.xf,
                clip: self.state.clip,
                stroke: Some((brush.into_owned(), width, &StrokeStyle::new())),
                ..Attrs::default()
            },
            self.decimal_digits,
        );
    }

    fn stroke_styled(
        &mut self,
        shape: impl Shape,
        brush: &impl IntoBrush<Self>,
        width: f64,
        style: &StrokeStyle,
    ) {
        let brush = brush.make_brush(self, || shape.bounding_box());
        add_shape(
            &mut self.doc,
            shape,
            &Attrs {
                xf: self.state.xf,
                clip: self.state.clip,
                stroke: Some((brush.into_owned(), width, style)),
                ..Attrs::default()
            },
            self.decimal_digits,
        );
    }

    fn text(&mut self) -> &mut Self::Text {
        &mut self.text
    }

    fn draw_text(&mut self, layout: &Self::TextLayout, pos: impl Into<Point>) {
        let pos = pos.into();

        let color = {
            let (r, g, b, a) = layout.text_color.as_rgba8();
            format!("rgba({}, {}, {}, {})", r, g, b, a as f64 * (100. / 255.))
        };

        let mut x = pos.x;
        // SVG doesn't do multiline text, and so doesn't have a concept of text width. We can do
        // alignment though, using text-anchor. TODO eventually we should generate a separate text
        // span for each line (having laid out the multiline text ourselves.
        let anchor = match (layout.max_width, layout.alignment) {
            (width, TextAlignment::End) if width.is_finite() && width > 0. => {
                x += width;
                "text-anchor:end"
            }
            (width, TextAlignment::Center) if width.is_finite() && width > 0. => {
                x += width * 0.5;
                "text-anchor:middle"
            }
            _ => "",
        };

        // If we are using a named font, then mark it for inclusion.
        self.text()
            .seen_fonts
            .lock()
            .unwrap()
            .insert(layout.font_face.clone());

        // We use the top of the text for y position, but SVG uses baseline, so we need to convert
        // between the two.
        //
        // `dominant-baseline` gets us most of the way (to the top of the ascender), so we add a
        // small fiddle factor in to cover the difference between the top of the line and the top
        // of the ascender (currently 6% of the font height, calcuated by eye).
        let y = pos.y + 0.06 * layout.size().height;
        let mut text = svg::node::element::Text::new()
            .set("x", x)
            .set("y", y)
            .set("dominant-baseline", "hanging")
            .set(
                "style",
                format!(
                    "font-size:{}pt;\
                        font-family:\"{}\";\
                        font-weight:{};\
                        font-style:{};\
                        text-decoration:{};\
                        fill:{};\
                        {}",
                    layout.font_size,
                    layout.font_face.family.name(),
                    layout.font_face.weight.to_raw(),
                    match layout.font_face.style {
                        FontStyle::Regular => "normal",
                        FontStyle::Italic => "italic",
                    },
                    match (layout.underline, layout.strikethrough) {
                        (false, false) => "none",
                        (false, true) => "line-through",
                        (true, false) => "underline",
                        (true, true) => "underline line-through",
                    },
                    color,
                    anchor,
                ),
            )
            .add(svg::node::Text::new(layout.text()));

        let affine = self.current_transform();
        if affine != Affine::IDENTITY {
            text.assign("transform", xf_val(&affine));
        }
        if let Some(id) = self.state.clip {
            text.assign("clip-path", format!("url(#{})", id.to_string()));
        }
        self.doc.append(text);
    }

    fn save(&mut self) -> Result<()> {
        let new = self.state.clone();
        self.stack.push(mem::replace(&mut self.state, new));
        Ok(())
    }

    fn restore(&mut self) -> Result<()> {
        self.state = self.stack.pop().ok_or(Error::StackUnbalance)?;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.doc.assign("x", self.bounds.x0);
        self.doc.assign("y", self.bounds.y0);
        self.doc.assign("width", self.bounds.width());
        self.doc.assign("height", self.bounds.height());
        self.doc.assign(
            "viewBox",
            (
                self.viewbox.x0,
                self.viewbox.y0,
                self.viewbox.width(),
                self.viewbox.height(),
            ),
        );
        self.doc.assign(
            "style",
            format!(
                "width:{}px;height:{}px;",
                self.viewbox.width(),
                self.viewbox.height()
            ),
        );

        let text = (*self.text()).clone();
        let mut seen_fonts = text.seen_fonts.lock().unwrap();
        if !seen_fonts.is_empty() {
            // include fonts
            let mut style = String::new();
            for face in &*seen_fonts {
                if face.family.name().contains('"') {
                    panic!("font family name contains `\"`");
                }
                // TODO convert font to woff2 to save space in svg output, maybe
                writeln!(
                    &mut style,
                    "@font-face {{\n\
                        font-family: \"{}\";\n\
                        font-weight: {};\n\
                        font-style: {};\n\
                        src: url(\"data:application/x-font-opentype;charset=utf-8;base64,{}\");\n\
                    }}",
                    face.family.name(),
                    face.weight.to_raw(),
                    match face.style {
                        FontStyle::Regular => "normal",
                        FontStyle::Italic => "italic",
                    },
                    base64::display::Base64Display::with_config(
                        &text.font_data(face)?,
                        base64::STANDARD
                    ),
                )
                .unwrap();
            }
            self.doc.append(svg::node::element::Style::new(style));
        }

        seen_fonts.clear();
        Ok(())
    }

    fn transform(&mut self, transform: Affine) {
        self.state.xf *= transform;
    }

    fn current_transform(&self) -> Affine {
        self.state.xf
    }

    fn make_image(
        &mut self,
        width: usize,
        height: usize,
        buf: &[u8],
        format: ImageFormat,
    ) -> Result<Self::Image> {
        Ok(SvgImage(match format {
            ImageFormat::Grayscale => {
                let image = ImageBuffer::from_raw(width as _, height as _, buf.to_owned())
                    .ok_or(Error::InvalidInput)?;
                DynamicImage::ImageLuma8(image)
            }
            ImageFormat::Rgb => {
                let image = ImageBuffer::from_raw(width as _, height as _, buf.to_owned())
                    .ok_or(Error::InvalidInput)?;
                DynamicImage::ImageRgb8(image)
            }
            ImageFormat::RgbaSeparate => {
                let image = ImageBuffer::from_raw(width as _, height as _, buf.to_owned())
                    .ok_or(Error::InvalidInput)?;
                DynamicImage::ImageRgba8(image)
            }
            ImageFormat::RgbaPremul => {
                use image::Rgba;
                use piet::util::unpremul;

                let mut image =
                    ImageBuffer::<Rgba<u8>, _>::from_raw(width as _, height as _, buf.to_owned())
                        .ok_or(Error::InvalidInput)?;
                for px in image.pixels_mut() {
                    px[0] = unpremul(px[0], px[3]);
                    px[1] = unpremul(px[1], px[3]);
                    px[2] = unpremul(px[2], px[3]);
                }
                DynamicImage::ImageRgba8(image)
            }
            // future-proof
            _ => return Err(Error::Unimplemented),
        }))
    }

    #[inline]
    fn draw_image(
        &mut self,
        image: &Self::Image,
        dst_rect: impl Into<Rect>,
        interp: InterpolationMode,
    ) {
        draw_image(self, image, None, dst_rect.into(), interp);
    }

    #[inline]
    fn draw_image_area(
        &mut self,
        image: &Self::Image,
        src_rect: impl Into<Rect>,
        dst_rect: impl Into<Rect>,
        interp: InterpolationMode,
    ) {
        draw_image(self, image, Some(src_rect.into()), dst_rect.into(), interp);
    }

    fn capture_image_area(&mut self, _src_rect: impl Into<Rect>) -> Result<Self::Image> {
        Err(Error::Unimplemented)
    }

    fn blurred_rect(&mut self, rect: Rect, _blur_radius: f64, brush: &impl IntoBrush<Self>) {
        // TODO blur (perhaps using SVG filters)
        self.fill(rect, brush)
    }
}

fn draw_image(
    ctx: &mut RenderContext,
    image: &<RenderContext as piet::RenderContext>::Image,
    _src_rect: Option<Rect>,
    dst_rect: Rect,
    _interp: InterpolationMode,
) {
    use image::ImageEncoder as _;

    let mut writer = base64::write::EncoderStringWriter::from(
        String::from("data:image/png;base64,"),
        base64::STANDARD,
    );

    image::codecs::png::PngEncoder::new(&mut writer)
        .write_image(
            image.0.as_bytes(),
            image.0.width(),
            image.0.height(),
            image.0.color(),
        )
        .unwrap();

    let data_url = writer.into_inner();

    // TODO when src_rect.is_some()
    // TODO maybe we could use css 'image-rendering' to control interpolation?
    let mut node = svg::node::element::Image::new()
        .set("x", dst_rect.x0)
        .set("y", dst_rect.y0)
        .set("width", dst_rect.x1 - dst_rect.x0)
        .set("height", dst_rect.y1 - dst_rect.y0)
        .set("href", data_url);

    let affine = piet::RenderContext::current_transform(ctx);
    if affine != Affine::IDENTITY {
        node.assign("transform", xf_val(&affine));
    }
    if let Some(id) = ctx.state.clip {
        node.assign("clip-path", format!("url(#{})", id.to_string()));
    }

    ctx.doc.append(node);
}

#[derive(Default)]
struct Attrs<'a> {
    xf: Affine,
    clip: Option<Id>,
    fill: Option<(Brush, Option<&'a str>)>,
    stroke: Option<(Brush, f64, &'a StrokeStyle)>,
}

impl Attrs<'_> {
    // allow clippy warning for `width != 1.0` in if statement
    #[allow(clippy::float_cmp)]
    fn apply_to(&self, node: &mut impl Node, decimal_digits: usize) {
        node.assign("transform", xf_val(&self.xf));
        if let Some(id) = self.clip {
            node.assign("clip-path", format!("url(#{})", id.to_string()));
        }
        if let Some((ref brush, rule)) = self.fill {
            node.assign("fill", brush.color());
            if let Some(opacity) = brush.opacity() {
                node.assign("fill-opacity", opacity);
            }
            if let Some(rule) = rule {
                node.assign("fill-rule", rule);
            }
        } else {
            node.assign("fill", "none");
        }
        if let Some((ref stroke, width, style)) = self.stroke {
            node.assign("stroke", stroke.color());
            if let Some(opacity) = stroke.opacity() {
                node.assign("stroke-opacity", opacity);
            }
            if width != 1.0 {
                node.assign("stroke-width", format!("{:.*}", decimal_digits, width));
            }
            match style.line_join {
                LineJoin::Miter { limit } if limit == LineJoin::DEFAULT_MITER_LIMIT => (),
                LineJoin::Miter { limit } => {
                    node.assign("stroke-miterlimit", limit);
                }
                LineJoin::Round => {
                    node.assign("stroke-linejoin", "round");
                }
                LineJoin::Bevel => {
                    node.assign("stroke-linejoin", "bevel");
                }
            }
            match style.line_cap {
                LineCap::Round => {
                    node.assign("stroke-linecap", "round");
                }
                LineCap::Square => {
                    node.assign("stroke-linecap", "square");
                }
                LineCap::Butt => (),
            }
            if !style.dash_pattern.is_empty() {
                node.assign("stroke-dasharray", style.dash_pattern.to_vec());
            }
            if style.dash_offset != 0.0 {
                node.assign("stroke-dashoffset", style.dash_offset);
            }
        }
    }
}

fn xf_val(xf: &Affine) -> svg::node::Value {
    let xf = xf.as_coeffs();
    format!(
        "matrix({} {} {} {} {} {})",
        xf[0], xf[1], xf[2], xf[3], xf[4], xf[5]
    )
    .into()
}

fn add_shape(node: &mut impl Node, shape: impl Shape, attrs: &Attrs, decimal_digits: usize) {
    if let Some(circle) = shape.as_circle() {
        let mut x = svg::node::element::Circle::new()
            .set("cx", format!("{:.*}", decimal_digits, circle.center.x))
            .set("cy", format!("{:.*}", decimal_digits, circle.center.y))
            .set("r", format!("{:.*}", decimal_digits, circle.radius));
        attrs.apply_to(&mut x, decimal_digits);
        node.append(x);
    } else if let Some(round_rect) = shape
        .as_rounded_rect()
        .filter(|r| r.radii().as_single_radius().is_some())
    {
        let mut x = svg::node::element::Rectangle::new()
            .set("x", format!("{:.*}", decimal_digits, round_rect.origin().x))
            .set("y", format!("{:.*}", decimal_digits, round_rect.origin().y))
            .set(
                "width",
                format!("{:.*}", decimal_digits, round_rect.width()),
            )
            .set(
                "height",
                format!("{:.*}", decimal_digits, round_rect.height()),
            )
            .set(
                "rx",
                format!(
                    "{:.*}",
                    decimal_digits,
                    round_rect.radii().as_single_radius().unwrap()
                ),
            )
            .set(
                "ry",
                format!(
                    "{:.*}",
                    decimal_digits,
                    round_rect.radii().as_single_radius().unwrap()
                ),
            );
        attrs.apply_to(&mut x, decimal_digits);
        node.append(x);
    } else if let Some(rect) = shape.as_rect() {
        let mut x = svg::node::element::Rectangle::new()
            .set("x", format!("{:.*}", decimal_digits, rect.origin().x))
            .set("y", format!("{:.*}", decimal_digits, rect.origin().y))
            .set("width", format!("{:.*}", decimal_digits, rect.width()))
            .set("height", format!("{:.*}", decimal_digits, rect.height()));
        attrs.apply_to(&mut x, decimal_digits);
        node.append(x);
    } else {
        let mut path = svg::node::element::Path::new().set(
            "d",
            bezpath_to_svg_w_decimal_digits(&shape.into_path(1e-3), decimal_digits),
        );
        attrs.apply_to(&mut path, decimal_digits);
        node.append(path);
    }
}

#[derive(Debug, Clone, Default)]
struct State {
    xf: Affine,
    clip: Option<Id>,
}

/// An SVG brush
#[derive(Debug, Clone)]
pub struct Brush {
    kind: BrushKind,
}

#[derive(Debug, Clone)]
enum BrushKind {
    Solid(Color),
    Ref(Id),
}

impl Brush {
    fn color(&self) -> svg::node::Value {
        match self.kind {
            BrushKind::Solid(color) => fmt_color(color).into(),
            BrushKind::Ref(id) => format!("url(#{})", id.to_string()).into(),
        }
    }

    fn opacity(&self) -> Option<svg::node::Value> {
        match self.kind {
            BrushKind::Solid(color) => Some(fmt_opacity(color).into()),
            BrushKind::Ref(_) => None,
        }
    }
}

impl IntoBrush<RenderContext> for Brush {
    fn make_brush<'b>(
        &'b self,
        _piet: &mut RenderContext,
        _bbox: impl FnOnce() -> Rect,
    ) -> Cow<'b, Brush> {
        Cow::Owned(self.clone())
    }
}

// RGB in hex representation
fn fmt_color(color: Color) -> String {
    format!("#{:06x}", color.as_rgba_u32() >> 8)
}

// Opacity as value from [0, 1]
fn fmt_opacity(color: Color) -> String {
    format!("{}", color.as_rgba().3)
}

#[derive(Clone)]
pub struct SvgImage(image::DynamicImage);

impl Image for SvgImage {
    fn size(&self) -> Size {
        let (width, height) = self.0.dimensions();
        Size {
            width: width as _,
            height: height as _,
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct Id(u64);

impl Id {
    // TODO allowing clippy warning temporarily. But this should be changed to impl Display
    #[allow(clippy::inherent_to_string)]
    fn to_string(self) -> String {
        const ALPHABET: &[u8; 52] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let mut out = String::with_capacity(4);
        let mut x = self.0;
        loop {
            let digit = (x % ALPHABET.len() as u64) as usize;
            out.push(ALPHABET[digit] as char);
            x /= ALPHABET.len() as u64;
            if x == 0 {
                break;
            }
        }
        out
    }
}

impl From<Id> for svg::node::Value {
    fn from(x: Id) -> Self {
        x.to_string().into()
    }
}

fn bezpath_to_svg_w_decimal_digits(bez_path: &BezPath, decimal_digits: usize) -> String {
    let mut svg_string = String::new();

    for el in bez_path.elements() {
        match *el {
            PathEl::MoveTo(p) => write!(
                svg_string,
                "M{} {}",
                format!("{:.*}", decimal_digits, p.x),
                format!("{:.*}", decimal_digits, p.y)
            )
            .unwrap(),
            PathEl::LineTo(p) => write!(
                svg_string,
                "L{} {}",
                format!("{:.*}", decimal_digits, p.x),
                format!("{:.*}", decimal_digits, p.y)
            )
            .unwrap(),
            PathEl::QuadTo(p1, p2) => write!(
                svg_string,
                "Q{} {} {} {}",
                format!("{:.*}", decimal_digits, p1.x),
                format!("{:.*}", decimal_digits, p1.y),
                format!("{:.*}", decimal_digits, p2.x),
                format!("{:.*}", decimal_digits, p2.y)
            )
            .unwrap(),
            PathEl::CurveTo(p1, p2, p3) => write!(
                svg_string,
                "C{} {} {} {} {} {}",
                format!("{:.*}", decimal_digits, p1.x),
                format!("{:.*}", decimal_digits, p1.y),
                format!("{:.*}", decimal_digits, p2.x),
                format!("{:.*}", decimal_digits, p2.y),
                format!("{:.*}", decimal_digits, p3.x),
                format!("{:.*}", decimal_digits, p3.y)
            )
            .unwrap(),
            PathEl::ClosePath => write!(svg_string, "Z").unwrap(),
        };
    }

    svg_string
}
