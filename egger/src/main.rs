// Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use clap::{App, Arg};
use colored::Colorize;
use egg::*;
use itertools::{enumerate, Itertools};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::fs::*;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use std::usize;
use egger::metadata::DataKind;
use egger::model::*;
use egger::load::{load_computations, load_eclasses, zip_computation_to_eclass};
use egger::rewrite::elementary_lemma;
use egger::rewrite::hack_lemma;
use egger::rewrite::inverse_lemma;
use egger::rewrite::lambda_rewrite;
use egger::rewrite::mask_lemma;
use egger::rewrite::noncompute_lemma;
use egger::rewrite::norm_activation_lemma;
use egger::rewrite::prune_lemma;
use egger::rewrite::reshape_lemma;
use egger::special::hlo_lemma;
use egger::special::special_lemma;
use egger::special::vllm_lemma;
use egger::symval::{load_parse_add, ShapeLike, SymValManager, SymValManagerRef};
use egger::utils::{shape_to_underscore_name, shapelike_name_to_vec, SEXPR_SPLIT};

fn main() -> ExitCode {
    let main_begin_time = Instant::now();
    // Parse arguments
    let matches = App::new("TGSaturate")
        .arg(
            Arg::with_name("mode")
                .short("m")
                .long("mode")
                .takes_value(true)
                .default_value("saturate")
                .help("Mode to run, can be saturate_precondition or check_impl"),
        )
        .arg(
            Arg::with_name("out_file")
                .short("o")
                .long("out_file")
                .takes_value(true)
                .help("Provide a output file name. For mode convert, it's for converted rules; for mode optimize, it's for measured runtime"),
        )
        .arg(
            Arg::with_name("input_dirname")
                .long("input_dirname")
                .takes_value(true)
                .default_value("target")
                .help("The name of precondition directory containing cs.sexpr, cd.sexpr and precondition.sexpr."),
        )
        .arg(
            Arg::with_name("saturated_path")
                .long("saturated_path")
                .takes_value(true)
                .default_value("target/saturated.json")
                .help("The path to output saturated files."),
        )
        .arg(
            Arg::with_name("n_iter")
                .long("n_iter")
                .takes_value(true)
                .default_value("25")
                .help("Max number of iterations for egg to run"),
        )
        .arg(
            Arg::with_name("n_sec")
                .long("n_sec")
                .takes_value(true)
                .default_value("3600")
                .help("Max number of seconds for egg to run"),
        )
        .arg(
            Arg::with_name("n_nodes")
                .long("n_nodes")
                .takes_value(true)
                .default_value("200000")
                .help("Max number of nodes for egraph"),
        )
        .arg(
            Arg::with_name("explanation")
                .long("explanation")
                .help("Enable egg's explanation."),
        )
        .arg(
            Arg::with_name("verbose")
                .long("verbose")
                .help("Print verbose information."),
        )
        .arg(
            Arg::with_name("visualize")
                .long("visualize")
                .help("Visualize start and saturated graph."),
        )
        .arg(
            Arg::with_name("stats")
                .long("stats")
                .help("Enable statistics."),
        )
        .arg(
            Arg::with_name("inverse_lemma")
                .long("inverse_lemma")
                .help("Use inverse lemmas."),
        )
        .get_matches();

    let run_mode = matches.value_of("mode").unwrap();
    let verbose = matches.is_present("verbose");
    let visualize = matches.is_present("visualize");
    let stats = matches.is_present("stats");

    // Global SymValManager
    let symval_manager = Arc::new(RwLock::new(SymValManager::new(
        "default_symval_manager",
        verbose,
    )));

    // Load SymVal preconditions into smtlib.
    let begin = Instant::now();
    let precondition_dirname = Path::new(matches.value_of("input_dirname").unwrap());
    let symval_precondition_path = Path::join(
        precondition_dirname,
        Path::new("precondition.scalar.smtlib"),
    )
    .to_path_buf();
    if Path::exists(symval_precondition_path.as_path()) {
        load_parse_add(&symval_precondition_path, symval_manager.clone());
    }
    println!("Loaded symval in {:?}", begin.elapsed());

    // Load name to shape mapping
    let shape_path = Path::join(precondition_dirname, Path::new("shapes.json"));
    let name_to_shapes: HashMap<String, ShapeLike> = {
        let content = std::fs::read_to_string(shape_path)
            .expect(format!("Something went wrong reading the shapes file").as_str());
        serde_json::from_str::<HashMap<String, String>>(&content)
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), shapelike_name_to_vec(&v, symval_manager.clone())))
            .collect()
    };

    // Analaysis
    let analysis = TensorAnalysis {
        manager: Some(symval_manager.clone()),
        name_to_shapes: name_to_shapes,
        enable_stats: stats,
        ..Default::default()
    };

    // Setup rules
    let rules = match run_mode {
        "saturate_precondition" | "check_impl" => {
            let mut rules = vec![];
            rules.extend(norm_activation_lemma::get_rules(
                symval_manager.clone(),
                verbose,
            ));
            rules.extend(elementary_lemma::get_rules(symval_manager.clone(), verbose));
            rules.extend(hack_lemma::get_rules(symval_manager.clone(), verbose));
            rules.extend(hlo_lemma::get_rules(symval_manager.clone(), verbose));
            rules.extend(lambda_rewrite::get_rules(symval_manager.clone(), verbose));
            rules.extend(mask_lemma::get_rules(symval_manager.clone(), verbose));
            rules.extend(noncompute_lemma::get_rules(symval_manager.clone(), verbose));
            rules.extend(reshape_lemma::get_rules(symval_manager.clone(), verbose));
            rules.extend(special_lemma::get_rules(symval_manager.clone(), verbose));
            rules.extend(vllm_lemma::get_rules(symval_manager.clone(), verbose));
            let use_inverse_lemmas = matches.is_present("inverse_lemma");
            if use_inverse_lemmas {
                rules.extend(inverse_lemma::get_rules(symval_manager.clone(), verbose));
            }
            rules
        }
        "self_provable" => {
            let mut rules = vec![];
            // rules.extend(mask_lemma::get_rules(symval_manager.clone(), verbose));
            rules.extend(noncompute_lemma::get_rules(symval_manager.clone(), verbose));
            rules.extend(prune_lemma::get_rules(symval_manager.clone(), verbose));
            rules.extend(reshape_lemma::get_rules(symval_manager.clone(), verbose));
            rules
        }
        _ => panic!("Running mode not supported"),
    };
    let rules = lambda_rewrite::collect_lambda_rewrites_to_egg_rewrites(rules, None);

    let res = match run_mode {
        "saturate_precondition" => {
            saturate_precondition(matches, rules, analysis, symval_manager, visualize)
        }
        "check_impl" => check_impl(matches, rules, analysis, symval_manager, visualize),
        "self_provable" => self_provable(matches, rules, analysis, symval_manager, visualize),
        _ => panic!("Running mode not supported"),
    };
    println!("Egg total time: {:?}", main_begin_time.elapsed());
    res
}

fn saturate_precondition(
    matches: clap::ArgMatches,
    rules: Vec<Rewrite<Mdl, TensorAnalysis>>,
    analysis: TensorAnalysis,
    symval_manager: SymValManagerRef,
    visualize: bool,
) -> ExitCode {
    env_logger::init();

    let explanation = matches.is_present("explanation");
    let n_sec = matches.value_of("n_sec").unwrap().parse::<u64>().unwrap();
    let time_limit_sec = Duration::new(n_sec, 0);

    let precondition_dirname = Path::new(matches.value_of("input_dirname").unwrap());
    let saturated_path = Path::new(matches.value_of("saturated_path").unwrap());

    // Run saturation
    let iter_limit = matches
        .value_of("n_iter")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let node_limit = matches
        .value_of("n_nodes")
        .unwrap()
        .parse::<usize>()
        .unwrap();

    let mut runner = Runner::<Mdl, TensorAnalysis, ()>::new(analysis)
        .with_node_limit(node_limit)
        .with_time_limit(time_limit_sec)
        .with_iter_limit(iter_limit);
    if explanation {
        runner = runner.with_explanations_enabled();
    }

    let mut eclasses: Vec<(Vec<Id>, &str)> = Vec::new();

    // Add preconditions
    let preconditions = load_eclasses(&precondition_dirname.join("precondition.sexpr"));
    add_mdl_eclasses(preconditions, &mut runner, &mut eclasses, "<precondition>");

    // Add scalar preconditions
    let scalars = load_eclasses(&precondition_dirname.join("precondition.scalar.sexpr"));
    add_mdl_eclasses(scalars, &mut runner, &mut eclasses, "<precondition-scalar>");

    // Add cs
    let (cs, ys) = load_computations(&precondition_dirname.join("cs.sexpr"));
    let cs_eclasses = zip_computation_to_eclass(cs.clone(), ys.clone());
    add_mdl_eclasses(cs_eclasses, &mut runner, &mut eclasses, "<cs>");

    // Add cds
    let (cds, yis) = load_computations(&precondition_dirname.join("cd.sexpr"));
    let cd_eclasses = zip_computation_to_eclass(cds.clone(), yis.clone());
    add_mdl_eclasses(cd_eclasses, &mut runner, &mut eclasses, "<cds>");

    // Union and noop
    union_eclasses(&mut runner, eclasses);
    runner.egraph.rebuild();

    // Dump the start e-graph.
    println!("Dumping start e-graph...");
    println!("{:?}", runner.egraph.dump());
    let start_path = Path::join(precondition_dirname, Path::new("start.json"));
    dump_egraph_for_extractor(
        &runner.egraph,
        Some(&ys[0]),
        Some(&yis),
        symval_manager.clone(),
        &start_path,
    );
    if visualize {
        py_visualize(&start_path);
    }

    // Start saturation.
    println!("Starting saturation...");
    let start_time = Instant::now();
    let runner = runner.run(&rules[..]);
    println!("Runner complete, taken {:?}", start_time.elapsed());
    print_runner_stats(&runner);

    let saturated_path = Path::new(saturated_path);
    dump_egraph_for_extractor(
        &runner.egraph,
        Some(&ys[0]),
        Some(&yis),
        symval_manager.clone(),
        saturated_path,
    );

    ExitCode::SUCCESS
}

fn check_impl(
    matches: clap::ArgMatches,
    rules: Vec<Rewrite<Mdl, TensorAnalysis>>,
    analysis: TensorAnalysis,
    symval_manager: SymValManagerRef,
    visualize: bool,
) -> ExitCode {
    env_logger::init();

    let explanation = matches.is_present("explanation");

    let input_dirname = Path::new(matches.value_of("input_dirname").unwrap());
    let saturated_path = Path::new(matches.value_of("saturated_path").unwrap());

    let n_sec = matches.value_of("n_sec").unwrap().parse::<u64>().unwrap();
    let time_limit_sec = Duration::new(n_sec, 0);

    // Run saturation
    let iter_limit = matches
        .value_of("n_iter")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let node_limit = matches
        .value_of("n_nodes")
        .unwrap()
        .parse::<usize>()
        .unwrap();

    let mut runner = Runner::<Mdl, TensorAnalysis, ()>::new(analysis)
        .with_node_limit(node_limit)
        .with_time_limit(time_limit_sec)
        .with_iter_limit(iter_limit);
    if explanation {
        runner = runner.with_explanations_enabled();
    }

    let mut eclasses: Vec<(Vec<Id>, &str)> = Vec::new();
    // Precondition
    let precondition_path = input_dirname.join("precondition.sexpr");
    load_sexpr(
        &precondition_path,
        &mut runner,
        &mut eclasses,
        "<precondition>",
    );
    // Computation of Single-device Graphs
    let cs_path = input_dirname.join("cs.sexpr");
    if cs_path.exists() {
        load_sexpr(&cs_path, &mut runner, &mut eclasses, "<cs>");
    }
    // Computation of Distributed Graphs
    let cd_path = input_dirname.join("cd.sexpr");
    if cd_path.exists() {
        load_sexpr(&cd_path, &mut runner, &mut eclasses, "<cd>");
    }
    // Computation of Expection.
    let ce_path = input_dirname.join("ce.sexpr");
    if ce_path.exists() {
        load_sexpr(&ce_path, &mut runner, &mut eclasses, "<ce>");
    }
    // Expected.
    let expected_equivalent: Vec<Vec<RecExpr<Mdl>>> =
        load_eclasses(&input_dirname.join("impl_checklist.sexpr"));

    // Union and noop
    union_eclasses(&mut runner, eclasses);
    runner.egraph.rebuild();

    // Dump the start e-graph.
    println!("Dumping start e-graph...");
    println!("{:?}", runner.egraph.dump());
    let start_path = Path::join(input_dirname, Path::new("start.json"));
    if visualize {
        dump_egraph_for_extractor(
            &runner.egraph,
            None,
            None,
            symval_manager.clone(),
            &start_path,
        );
        py_visualize(&start_path);
    }

    // Start saturation.
    println!("Starting saturation...");
    let start_time = Instant::now();
    let runner = runner.run(&rules[..]);
    println!("Runner complete, taken {:?}", start_time.elapsed());
    print_runner_stats(&runner);

    let saturated_path = Path::new(saturated_path).to_path_buf();
    if visualize {
        dump_egraph_for_extractor(
            &runner.egraph,
            None,
            None,
            symval_manager.clone(),
            &saturated_path,
        );
        py_visualize(&saturated_path);
    }

    let mut failed = false;
    for (idx, expected) in enumerate(expected_equivalent) {
        println!(
            "---------- Checking equivalence {} ----------",
            format!("group: {idx}").bold()
        );
        let mut ids: HashSet<Id> = HashSet::new();
        for recexpr in expected {
            let id = runner.egraph.lookup_expr(&recexpr).unwrap();
            println!("{}: {}", recexpr.pretty(usize::MAX), id);
            ids.insert(id);
        }
        if ids.len() == 1 {
            println!("{}", "Passed equivalence checking.".green().bold());
        } else {
            println!("{}", "Failed equivalence checking.".red().bold());
            failed = true;
        }
        println!("---------------------------------------------------\n");
    }
    if failed {
        return ExitCode::FAILURE;
    } else {
        return ExitCode::SUCCESS;
    }
}

fn self_provable(
    matches: clap::ArgMatches,
    rules: Vec<Rewrite<Mdl, TensorAnalysis>>,
    analysis: TensorAnalysis,
    symval_manager: SymValManagerRef,
    visualize: bool,
) -> ExitCode {
    env_logger::init();

    let explanation = matches.is_present("explanation");

    let input_dirname = Path::new(matches.value_of("input_dirname").unwrap());
    let saturated_path = Path::new(matches.value_of("saturated_path").unwrap());

    let n_sec = matches.value_of("n_sec").unwrap().parse::<u64>().unwrap();
    let time_limit_sec = Duration::new(n_sec, 0);

    let mut runner =
        Runner::<Mdl, TensorAnalysis, ()>::new(analysis).with_time_limit(time_limit_sec);
    if explanation {
        runner = runner.with_explanations_enabled();
    }

    // Expected.
    let content = std::fs::read_to_string(input_dirname.join("impl_checklist.sexpr"))
        .expect("Something went wrong reading the impl_checklist file");
    let expected_equivalent: Vec<Vec<RecExpr<Mdl>>> = content
        .split("\n")
        .filter(|line| line.len() > 0)
        .map(|line| {
            let eclass = line
                .split(SEXPR_SPLIT)
                .map(|sexpr| sexpr.parse().unwrap())
                .collect_vec();
            for sexpr in &eclass {
                let id = runner.egraph.add_expr(sexpr);
                print!("\tAdded ({}): {:?}\n", id, sexpr.pretty(usize::MAX));
            }
            eclass
        })
        .collect_vec();

    runner.egraph.rebuild();

    // Dump the start e-graph.
    println!("Dumping start e-graph...");
    println!("{:?}", runner.egraph.dump());
    let start_path = Path::join(input_dirname, Path::new("start.json"));
    if visualize {
        dump_egraph_for_extractor(
            &runner.egraph,
            None,
            None,
            symval_manager.clone(),
            &start_path,
        );
        py_visualize(&start_path);
    }

    // Start saturation.
    println!("Starting saturation...");
    let start_time = Instant::now();
    let runner = runner.run(&rules[..]);
    println!("Runner complete, taken {:?}", start_time.elapsed());
    print_runner_stats(&runner);

    let saturated_path = Path::new(saturated_path).to_path_buf();
    if visualize {
        dump_egraph_for_extractor(
            &runner.egraph,
            None,
            None,
            symval_manager.clone(),
            &saturated_path,
        );
        py_visualize(&saturated_path);
    }

    for (idx, expected) in enumerate(expected_equivalent) {
        println!(
            "---------- Checking Self Provable {} ----------",
            format!("group: {idx}").bold()
        );
        let mut class_ids: Vec<Id> = Vec::new();
        let mut ids: HashSet<Id> = HashSet::new();
        for recexpr in expected {
            let id = runner.egraph.lookup_expr(&recexpr).unwrap();
            println!("{}: {}", recexpr.pretty(usize::MAX), id);
            ids.insert(id);
            class_ids.push(id);
        }
        println!("{}", "Done self provable checking.".green().bold());
        write(
            Path::join(input_dirname, Path::new("result_ids.txt")),
            class_ids.iter().map(|id| id.to_string()).join(","),
        )
        .expect("Unable to write file");
        println!("---------------------------------------------------\n");
    }
    return ExitCode::SUCCESS;
}

fn union_eclasses(runner: &mut Runner<Mdl, TensorAnalysis, ()>, eclasses: Vec<(Vec<Id>, &str)>) {
    // Union equivalent nodes before doing saturation
    for eclass in eclasses {
        let reason = eclass.1;
        let mut current_id = &eclass.0[0];
        for next_id in &eclass.0[1..] {
            runner.egraph.union_trusted(*current_id, *next_id, reason);
            current_id = next_id;
        }
    }
}

fn dump_egraph_for_extractor(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    y: Option<&RecExpr<Mdl>>,
    yis: Option<&Vec<RecExpr<Mdl>>>,
    symval_manager: SymValManagerRef,
    path: &Path,
) {
    let class_ids: Vec<Id> = egraph.classes().map(|c| egraph.find(c.id)).collect();
    assert!(class_ids.len() == egraph.number_of_classes());
    let id_to_idx_map: HashMap<Id, usize> = class_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    let num_classes = egraph.number_of_classes();
    let num_nodes = egraph.total_size();
    let mut nodes: Vec<(String, usize, Vec<usize>)> = Vec::new();
    let mut classes: Vec<(usize, Vec<usize>, String)> =
        vec![(0, Vec::new(), "".to_string()); num_classes];
    let mut eclass_of_node: Vec<usize> = Vec::with_capacity(num_nodes);

    let mut i = 0;

    let manager = symval_manager.read().unwrap();
    let eclasses = egraph.classes().sorted_by_key(|eclass| eclass.id);
    for class in eclasses {
        let idx = *id_to_idx_map.get(&egraph.find(class.id)).unwrap();
        let class_id = class.id.into();
        for node in class.iter() {
            let mut node_string = node.to_string();
            if let Some(int_expr) = manager.int_map.get(&node_string) {
                node_string = int_expr.to_string();
            }
            nodes.push((
                node_string,
                class_id,
                node.children().iter().map(|&id| id.into()).collect(),
            ));
            classes[idx].0 = class_id;
            classes[idx].1.push(i);
            if class.data.dtype == DataKind::Tnsr {
                classes[idx].2 = shape_to_underscore_name(&class.data.get_shape());
            }
            eclass_of_node.push(class_id);
            i += 1;
        }
    }

    // let root_m = *id_m_map.get(&egraph.find(root)).unwrap();
    let data = json!({
        "nodes": nodes,
        "classes": classes,
        "eclass_of_node": eclass_of_node,
        "y": if y.is_some() {y.unwrap().pretty(usize::MAX)} else {"".to_string()},
        "yis": if yis.is_some() {yis.unwrap().iter().map(|yi| yi.pretty(usize::MAX)).collect::<Vec<String>>()} else {vec![]},
    });
    let data_str = serde_json::to_string(&data).expect("Fail to convert json to string");
    write(path, data_str).expect("Unable to write file");
}

fn print_runner_stats(runner: &Runner<Mdl, TensorAnalysis, ()>) {
    println!("  Stopped: {:?}", runner.stop_reason.as_ref().unwrap());
    println!("  Number of iterations: {:?}", runner.iterations.len() - 1);
    let (num_enodes, num_classes, avg_nodes_per_class, num_edges, num_programs) =
        get_stats(&runner.egraph);
    println!("  Number of enodes: {}", num_enodes);
    println!("  Number of classes: {}", num_classes);
    println!("  Average nodes per class: {}", avg_nodes_per_class);
    println!("  Number of edges: {}", num_edges);
    println!("  Number of programs: {}", num_programs);

    println!("Lemma applied count:");
    for (lemma_name, count) in &runner.egraph.analysis.lemma_applied_count {
        if *count > 0 {
            println!("Lemma {} was applied {} times.", lemma_name, count);
        }
    }

    println!("Result egraph:\n{:?}", runner.egraph.dump());
}

pub fn add_mdl_eclasses<'a>(
    mdl_eclasses: Vec<Vec<RecExpr<Mdl>>>,
    runner: &mut Runner<Mdl, TensorAnalysis>,
    eclasses: &mut Vec<(Vec<Id>, &'a str)>,
    reason: &'a str,
) {
    for (idx, mdl_eclass) in mdl_eclasses.iter().enumerate() {
        eclasses.push({
            println!(
                "{}[{}]: {}",
                reason,
                idx,
                mdl_eclass
                    .iter()
                    .map(|x| x.pretty(usize::MAX))
                    .collect_vec()
                    .join(", ")
            );
            let mut id_eclass = vec![];
            for sexpr in mdl_eclass {
                let sexpr_id = runner.egraph.add_expr(sexpr);
                id_eclass.push(sexpr_id);
            }
            (id_eclass, reason)
        });
    }
}

pub fn load_sexpr<'a>(
    path: &PathBuf,
    runner: &mut Runner<Mdl, TensorAnalysis>,
    eclasses: &mut Vec<(Vec<Id>, &'a str)>,
    reason: &'a str,
) {
    let loaded_eclasses = load_eclasses(path);
    add_mdl_eclasses(loaded_eclasses, runner, eclasses, reason);
}

pub fn py_visualize(path: &PathBuf) -> ExitCode {
    // Visualize the egraph using the extractor.
    std::process::Command::new("python")
        .arg("extractor/visualize.py")
        .arg("-i")
        .arg(path.to_str().unwrap())
        .output()
        .expect("Failed to run extractor");
    ExitCode::SUCCESS
}

////////////////////////////////////////////////////////////////////////
/// Below are some unused things (but maybe useful in the future).
////////////////////////////////////////////////////////////////////////

/// This function gets the following stats:
///     Total number of enodes
///     Total number of eclasses
///     Average number of enodes per class
///     Total number of edges (children relationships)
///     Total number of equivalent programs represented (power of 2)
fn get_stats(egraph: &EGraph<Mdl, TensorAnalysis>) -> (usize, usize, f32, usize, f32) {
    let num_enodes = egraph.total_size();
    let num_classes = egraph.number_of_classes();
    let avg_nodes_per_class = num_enodes as f32 / (num_classes as f32);
    let num_edges = egraph
        .classes()
        .fold(0, |acc, c| c.iter().fold(0, |sum, n| n.len() + sum) + acc);
    let num_programs = egraph
        .classes()
        .fold(0.0, |acc, c| acc + (c.len() as f32).log2());
    (
        num_enodes,
        num_classes,
        avg_nodes_per_class,
        num_edges,
        num_programs,
    )
}
