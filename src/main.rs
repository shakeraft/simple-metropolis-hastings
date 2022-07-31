use plotters::prelude::*;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, NormalError};

fn random_normal() -> Result<f32, NormalError> {
    let mut rng: ThreadRng = rand::thread_rng();
    let normal: Normal<f32> = Normal::new(0.0, 1.0)?;
    let value: f32 = normal.sample(&mut rng);
    Ok(value)
}

fn random_uniform() -> f32 {
    let mut rng: ThreadRng = rand::thread_rng();
    let value: f32 = rng.gen_range(0.0..1.0);
    value
}

fn metropolis_hastings_step(x_initial: f32, target_distribution: fn(f32) -> f32) -> (f32, i32) {
    // Generate candidate from Gaussian distibution
    let x_candidate: f32 = random_normal().expect("Error occured when generating candidate value!");

    // Calculate acceptance probability
    let acceptance_probability: f32 =
        (target_distribution(x_candidate) / target_distribution(x_initial)).min(1.0);

    // Generate uniform random value
    let uniform_rand_val: f32 = random_uniform();

    // Accept or reject candidate
    if uniform_rand_val < acceptance_probability {
        (x_candidate, 1)
    } else {
        (x_initial, 0)
    }
}

fn metropolis_hastings(
    burn_in: i32,
    num_samples: i32,
    mut sample: f32,
    target_distribution: fn(f32) -> f32,
) -> (Vec<f32>, i32) {
    // Storage
    let mut markov_chain: Vec<f32> = Vec::new();
    let mut acceptance: i32 = 0;
    let mut acceptance_status: i32;

    // Metropolis-Hastings routine
    // Burn-in period to allow for samples to converge
    for _ in 0..burn_in {
        (_, _) = metropolis_hastings_step(sample, target_distribution);
    }

    for _ in 0..num_samples {
        (sample, acceptance_status) = metropolis_hastings_step(sample, target_distribution);
        acceptance += acceptance_status;
        markov_chain.push(sample);
    }

    (markov_chain, acceptance)
}

// Plot Reference:
// https://github.com/plotters-rs/plotters/blob/master/plotters/examples/normal-dist2.rs
fn draw_plot(trace: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    const OUT_FILE_NAME: &'static str = "sample-result.png";
    let root = BitMapBackend::new(OUT_FILE_NAME, (1024, 768)).into_drawing_area();

    root.fill(&WHITE)?;

    // Number of bins = x_coord range width / step size
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .caption("Simple Metropolis-Hastings Run", ("sans-serif", 30))
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 60)
        .set_label_area_size(LabelAreaPosition::Right, 60)
        .build_cartesian_2d(-3f32..3f32, 0f32..0.1)?
        .set_secondary_coord(
            (-3f32..3f32).step(0.2).use_round().into_segmented(),
            0u32..1500u32,
        );

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .y_label_formatter(&|y| format!("{:.0}%", *y * 100.0))
        .y_desc("Percentage")
        .draw()?;

    chart.configure_secondary_axes().y_desc("Count").draw()?;

    let actual = Histogram::vertical(chart.borrow_secondary())
        .style(BLUE.filled())
        .margin(3)
        .data(trace.iter().map(|x| (*x, 1)));

    chart
        .draw_secondary_series(actual)?
        .label("Observed")
        .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 10, y + 5)], BLUE.filled()));

    chart.configure_series_labels().draw()?;

    root.present().expect("Unable to write result to file!");
    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}

// Function to approximate
fn target_distribution(x: f32) -> f32 {
    (-x.powi(2)).exp() * (2.0 + (5.0 * x).sin() + (2.0 * x).sin())
}

fn main() {
    let num_samples: i32 = 6000;
    let (markov_chain, acceptance) =
        metropolis_hastings(2500, num_samples, 0.0, target_distribution);
    println!(
        "Acceptance Rate: {}",
        acceptance as f32 / num_samples as f32
    );
    draw_plot(markov_chain).expect("Error with drawing histogram!");
}
