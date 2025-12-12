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

use colored::Colorize;
use core::panic;
use smtlib::lowlevel::lexicon::Numeral;
use smtlib::terms::Dynamic;
use smtlib::{
    backend::z3_binary::Z3Binary, lowlevel::ast::Command, Bool, Int, SatResult, Solver, Sort,
};
use smtlib_lowlevel::ast::{Identifier, QualIdentifier, Term};
use smtlib_lowlevel::lexicon::Symbol;
use std::cmp::{Ordering, PartialEq, PartialOrd};
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, Sub};
use std::path::PathBuf;
use std::sync::{atomic::AtomicUsize, atomic::Ordering::Relaxed, Arc, RwLock};
use std::time::Instant;

macro_rules! cannot_compare_warn {
    ($x:expr, $y:expr, $msg:expr) => {
        println!(
            "{}",
            format!(
                "Warning: found SymVal that cannot be compared: \"{}\", \"{}\". {}",
                $x, $y, $msg
            )
            .red()
            .bold()
        );
    };
}

pub fn make_clean_int(name: &str) -> Int {
    Int::from(Term::Identifier(QualIdentifier::Identifier(
        Identifier::Simple(Symbol(name.into())),
    )))
}

#[derive(Debug, Clone, Hash)]
pub enum SymValId {
    Val(i64),
    Name(String), // A name refers to a Z3 Expr
}

impl SymValId {
    pub fn val(&self) -> i64 {
        match self {
            SymValId::Val(x) => *x,
            _ => panic!("Not a value"),
        }
    }

    pub fn name(&self) -> String {
        match self {
            SymValId::Name(x) => x.clone(),
            _ => panic!("Not a name"),
        }
    }

    pub fn str_val_or_name(&self) -> String {
        match self {
            SymValId::Val(x) => x.to_string(),
            SymValId::Name(x) => x.clone(),
        }
    }

    pub fn is_val(&self) -> bool {
        match self {
            SymValId::Val(_) => true,
            _ => false,
        }
    }
}

#[derive(Clone)]
pub struct SymVal {
    pub symval_id: SymValId,
    pub manager: SymValManagerRef,
}

impl std::fmt::Debug for SymVal {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.symval_id {
            SymValId::Val(x) => write!(f, "{}", x),
            SymValId::Name(x) => write!(f, "{}", x),
        }
    }
}

impl std::hash::Hash for SymVal {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.symval_id.hash(state);
    }
}

impl SymVal {
    pub fn new_val(val: i64, manager: SymValManagerRef) -> SymVal {
        SymVal {
            symval_id: SymValId::Val(val),
            manager: manager,
        }
    }

    pub fn new_name(name: String, manager: SymValManagerRef) -> SymVal {
        SymVal {
            symval_id: SymValId::Name(name),
            manager: manager.clone(),
        }
    }

    pub fn from_str(s: &str, manager: SymValManagerRef) -> SymVal {
        if let Ok(val) = s.parse::<i64>() {
            SymVal::new_val(val, manager)
        } else {
            assert!(!s.contains(","), "Comma not allowed in sym name: {}", s);
            SymVal::new_name(s.to_string(), manager)
        }
    }

    pub fn zero(manager: SymValManagerRef) -> SymVal {
        SymVal::new_val(0, manager)
    }

    pub fn one(manager: SymValManagerRef) -> SymVal {
        SymVal::new_val(1, manager)
    }

    pub fn val(&self) -> i64 {
        match &self.symval_id {
            SymValId::Val(x) => *x,
            _ => panic!("Not a value"),
        }
    }

    pub fn name(&self) -> String {
        match &self.symval_id {
            SymValId::Name(x) => x.clone(),
            _ => panic!("Not a name"),
        }
    }

    pub fn str_val_or_name(&self) -> String {
        match &self.symval_id {
            SymValId::Val(x) => x.to_string(),
            SymValId::Name(x) => x.clone(),
        }
    }

    pub fn get_smtlib_int(&self) -> Int {
        match &self.symval_id {
            SymValId::Val(x) => {
                if *x < 0 {
                    Int::from(0) - Int::from((-*x) as i64)
                } else {
                    Int::from(*x as i64)
                }
            }
            SymValId::Name(x) => {
                let manager = self.manager.try_read().unwrap();
                if let Some(z3int) = manager.get_by_name(x) {
                    z3int
                } else if x.starts_with("(") {
                    // May be an expression, try to parse.
                    // FIXME: Can be slow due to dynamically parsing.
                    let mut parser = SmtlibParser::new(x);
                    let parsed = parser.parse_dynamic().unwrap();
                    let expr: Term = parsed.into();
                    let expr: Int = expr.into();
                    expr
                } else {
                    make_clean_int(x)
                    // panic!(
                    //     "Cannot find Z3 Int for name: {}. NOTE: used names should have been added because we should provide the preconditions.",
                    //     x
                    // );
                }
            }
        }
    }

    pub fn is_val(&self) -> bool {
        match self.symval_id {
            SymValId::Val(_) => true,
            _ => false,
        }
    }
}

impl Display for SymVal {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.symval_id {
            SymValId::Val(x) => write!(f, "{}", x),
            SymValId::Name(x) => write!(f, "\"{}\"", x), // Need to quote to avoid confusion.
        }
    }
}

macro_rules! partial_ord_fn {
    ($fn:tt, $val_op:tt, $op:tt, $opposed_op:tt) => {
        fn $op(&self, other: &SymVal) -> bool {
            match (&self.symval_id, &other.symval_id) {
                (SymValId::Val(x), SymValId::Val(y)) => x $val_op y,
                (x, y) => {
                    let x = x.str_val_or_name();
                    let y = y.str_val_or_name();
                    if x == y {
                        return true;
                    }

                    let x = self.get_smtlib_int();
                    let y = other.get_smtlib_int();

                    let mut manager = self.manager.try_write().unwrap();
                    // Test if equal
                    let op_res = manager.check_sat(!x.$op(y));
                    match op_res {
                        SatResult::Sat => {
                            let opposed_op_res = manager.check_sat(!x.$opposed_op(y));
                            match opposed_op_res {
                                SatResult::Unsat => false,
                                _ => {
                                    cannot_compare_warn!(
                                        self,
                                        other,
                                        format!("In PartialOrd {}, testing {}=>{:?} and {}=>{:?} are all Sat, assuming false.",
                                            stringify!($fn), stringify!($opposed_op), op_res, stringify!($opposed_op), opposed_op_res)
                                    );
                                    false
                                }
                            }
                        }
                        SatResult::Unsat => true,
                        SatResult::Unknown => {
                            cannot_compare_warn!(
                                self,
                                other,
                                format!("In PartialOrd {}, testing {}=>{}, assuming false.", stringify!($fn), stringify!($op), op_res)
                            );
                            false
                        }
                    }
                }
            }
        }
    }
}

impl PartialOrd<SymVal> for SymVal {
    fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
        panic!("Shouldn't call!");
    }

    partial_ord_fn!(ge, >=, ge, lt);
    partial_ord_fn!(gt, >,  gt, le);
    partial_ord_fn!(le, <=, le, gt);
    partial_ord_fn!(lt, <, lt, ge);
}

macro_rules! partial_ord_i64_fn {
    ($fn:tt) => {
        fn $fn(&self, other: &i64) -> bool {
            let other = SymVal::new_val(*other, self.manager.clone());
            self.$fn(&other)
        }
    };
}

impl PartialOrd<i64> for SymVal {
    fn partial_cmp(&self, _other: &i64) -> Option<Ordering> {
        panic!("Shouldn't call!");
    }

    partial_ord_i64_fn!(ge);
    partial_ord_i64_fn!(gt);
    partial_ord_i64_fn!(le);
    partial_ord_i64_fn!(lt);
}

impl PartialEq<SymVal> for SymVal {
    fn eq(&self, other: &SymVal) -> bool {
        match (&self.symval_id, &other.symval_id) {
            (SymValId::Val(x), SymValId::Val(y)) => x == y,
            (x, y) => {
                let x = x.str_val_or_name();
                let y = y.str_val_or_name();
                if x == y {
                    return true;
                }

                let x = self.get_smtlib_int();
                let y = other.get_smtlib_int();

                let mut manager = self.manager.try_write().unwrap();
                match manager.check_sat(x._neq(y)) {
                    SatResult::Sat => false,
                    SatResult::Unsat => true,
                    SatResult::Unknown => {
                        cannot_compare_warn!(
                            self,
                            other,
                            "In PartialOrd<Eq>, Unknown, assuming false."
                        );
                        false
                    }
                }
            }
        }
    }
}

impl PartialEq<i64> for SymVal {
    fn eq(&self, other: &i64) -> bool {
        let other = SymVal::new_val(*other, self.manager.clone());
        self == &other
    }
}
impl PartialEq<i32> for SymVal {
    fn eq(&self, other: &i32) -> bool {
        let other = SymVal::new_val(*other as i64, self.manager.clone());
        self == &other
    }
}
impl PartialEq<usize> for SymVal {
    fn eq(&self, other: &usize) -> bool {
        let other = SymVal::new_val(*other as i64, self.manager.clone());
        self == &other
    }
}

impl Add<SymVal> for SymVal {
    type Output = SymVal;
    fn add(self, other: SymVal) -> SymVal {
        match (&self.symval_id, &other.symval_id) {
            (SymValId::Val(x), SymValId::Val(y)) => SymVal::new_val(x + y, self.manager),
            _ => {
                let x_int = self.get_smtlib_int();
                let y_int = other.get_smtlib_int();
                let res_int = x_int + y_int;
                let name = {
                    let mut manager = self.manager.try_write().unwrap();
                    // let res_int: Int = manager.solver.simplify(res_int.into()).unwrap().into();
                    manager.add(res_int)
                };
                SymVal::new_name(name, self.manager.clone())
            }
        }
    }
}

impl AddAssign<SymVal> for SymVal {
    fn add_assign(&mut self, other: SymVal) {
        *self = self.clone().add(other);
    }
}

impl Div<SymVal> for SymVal {
    type Output = SymVal;
    fn div(self, other: SymVal) -> SymVal {
        match (&self.symval_id, &other.symval_id) {
            (SymValId::Val(x), SymValId::Val(y)) => SymVal::new_val(x / y, self.manager),
            _ => {
                let x_int = self.get_smtlib_int();
                let y_int = other.get_smtlib_int();
                let res_int = x_int / y_int;
                let name = {
                    let mut manager = self.manager.try_write().unwrap();
                    // let res_int: Int = manager.solver.simplify(res_int.into()).unwrap().into();
                    manager.add(res_int)
                };
                SymVal::new_name(name, self.manager.clone())
            }
        }
    }
}

impl Mul<SymVal> for SymVal {
    type Output = SymVal;
    fn mul(self, other: SymVal) -> SymVal {
        match (&self.symval_id, &other.symval_id) {
            (SymValId::Val(x), SymValId::Val(y)) => SymVal::new_val(x * y, self.manager),
            _ => {
                let x_int = self.get_smtlib_int();
                let y_int = other.get_smtlib_int();
                let res_int = x_int * y_int;
                let name = {
                    let mut manager = self.manager.try_write().unwrap();
                    // let res_int: Int = manager.solver.simplify(res_int.into()).unwrap().into();
                    manager.add(res_int)
                };
                SymVal::new_name(name, self.manager.clone())
            }
        }
    }
}

impl Sub<SymVal> for SymVal {
    type Output = SymVal;
    fn sub(self, other: SymVal) -> SymVal {
        match (&self.symval_id, &other.symval_id) {
            (SymValId::Val(x), SymValId::Val(y)) => SymVal::new_val(x - y, self.manager),
            _ => {
                let x_int = self.get_smtlib_int();
                let y_int = other.get_smtlib_int();
                let res_int = x_int - y_int;
                let name = {
                    let mut manager = self.manager.try_write().unwrap();
                    // let res_int: Int = manager.solver.simplify(res_int.into()).unwrap().into();
                    manager.add(res_int)
                };
                SymVal::new_name(name, self.manager.clone())
            }
        }
    }
}

pub type ShapeLike = Vec<SymVal>;

pub struct SymValManager {
    pub name: String,
    pub int_map: HashMap<String, Int>,
    pub int_str_to_name: HashMap<String, String>,
    pub int_name_vec: Vec<(Int, String)>,
    pub solver: Solver<Z3Binary>, // This is only used to simplify
    pub next_id: AtomicUsize,
    pub verbose: bool,
}

impl Clone for SymValManager {
    fn clone(&self) -> Self {
        SymValManager {
            name: self.name.clone(),
            int_map: self.int_map.clone(),
            int_str_to_name: self.int_str_to_name.clone(),
            int_name_vec: self.int_name_vec.clone(),
            solver: Solver::new(Z3Binary::new("z3").unwrap()).unwrap(),
            next_id: AtomicUsize::new(self.next_id.load(Relaxed)),
            verbose: self.verbose,
        }
    }
}

impl std::fmt::Debug for SymValManager {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SymValManager({})", self.name)
    }
}

impl std::hash::Hash for SymValManager {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl SymValManager {
    pub fn new(name: &str, verbose: bool) -> SymValManager {
        SymValManager {
            name: name.to_string(),
            int_map: HashMap::new(),
            int_str_to_name: HashMap::new(),
            int_name_vec: vec![],
            solver: Solver::new(Z3Binary::new("z3").unwrap()).unwrap(),
            next_id: AtomicUsize::new(2000),
            verbose: verbose,
        }
    }

    pub fn get_by_name(&self, name: &str) -> Option<Int> {
        match self.int_map.get(name) {
            Some(x) => Some(x.clone()),
            None => None,
        }
    }

    pub fn get_by_int(&self, val: Int) -> Option<Int> {
        match self.int_map.get(&val.to_string()) {
            Some(x) => Some(x.clone()),
            None => None,
        }
    }

    pub fn add(&mut self, val: Int) -> String {
        let simplified_val: Int = self.solver.simplify(val.clone().into()).unwrap().into();
        if let Some(name) = self.int_str_to_name.get(&simplified_val.to_string()) {
            name.clone()
        // } else if let Some(name) = self.get_equivalent_int_name(simplified_val.clone()) {
        //     name
        } else {
            let name = simplified_val.to_string();
            self.add_with_name(val, name.clone(), Some(simplified_val));
            name
        }
    }

    pub fn get_equivalent_int_name(&mut self, val: Int) -> Option<String> {
        let to_iter = self.int_name_vec.clone();
        for (int, name) in to_iter {
            match self.check_sat(!int._eq(val.clone())) {
                SatResult::Unsat => {
                    return Some(name.clone());
                }
                _ => {}
            }
        }
        None
    }

    pub fn add_with_name(&mut self, val: Int, name: String, simplified_val: Option<Int>) {
        self.int_map.insert(name.clone(), val.clone());
        let used_simplified_val = if let Some(simplified_val) = simplified_val {
            simplified_val
        } else {
            self.solver.simplify(val.clone().into()).unwrap().into()
        };
        self.int_str_to_name
            .insert(used_simplified_val.to_string(), name.clone());
        self.int_name_vec.push((used_simplified_val, name.clone()));
        if self.verbose {
            println!(
                "Added {}: val={}, simplified_val={:?}, used_simplified_val={}",
                name, val, simplified_val, used_simplified_val
            );
        }
    }

    pub fn check_sat(&mut self, to_check: Bool) -> SatResult {
        // Use Push/Pop to avoid re-adding preconditions.
        self.solver
            .run_command(&Command::Push(Numeral("1".to_string())))
            .unwrap();
        self.solver.assert(to_check).unwrap();
        let check_result = self.solver.check_sat();
        self.solver
            .run_command(&Command::Pop(Numeral("1".to_string())))
            .unwrap();
        check_result.unwrap()
    }
}

pub type SymValManagerRef = Arc<RwLock<SymValManager>>;

unsafe impl Send for SymValManager {}

unsafe impl Sync for SymValManager {}

pub struct SmtlibParser {
    pub s: String,
    pub op_stack: Vec<String>,
    pub operand_stack: Vec<Dynamic>,
    pub i: usize,
    pub tokens: Vec<String>,
    pub leaf_names: HashSet<String>,
}

impl SmtlibParser {
    fn new(s: &str) -> SmtlibParser {
        let mut tokens = vec![];
        let mut i = 0;
        while i < s.len() {
            while &s[i..i + 1] == " " {
                i += 1;
            }
            let mut j = i;
            if &s[i..i + 1] == "(" {
                tokens.push("(".to_string());
                i += 1;
            } else if &s[i..i + 1] == ")" {
                tokens.push(")".to_string());
                i += 1;
            } else {
                while j < s.len() {
                    let char_j = &s[j..j + 1];
                    if char_j == " " || char_j == "(" || char_j == ")" {
                        break;
                    }
                    j += 1;
                }
                tokens.push(s[i..j].to_string());
                i = j;
            }
        }
        SmtlibParser {
            s: s.to_string(),
            op_stack: vec![],
            operand_stack: vec![],
            i: 0,
            tokens: tokens,
            leaf_names: HashSet::new(),
        }
    }

    fn next_token(&mut self) -> String {
        self.i += 1;
        self.tokens[self.i - 1].to_string()
    }

    pub fn parse_dynamic(&mut self) -> Option<Dynamic> {
        while self.i < self.tokens.len() {
            let token = self.next_token();
            if token.starts_with("Sym") {
                self.leaf_names.insert(token.clone());
                self.operand_stack.push(make_clean_int(&token).into());
            } else if token == "(" {
                let token = self.next_token();
                if token.starts_with("Sym") {
                    return Some(make_clean_int(&token).into());
                } else {
                    self.op_stack.push(token);
                }
            } else if token == ")" {
                let op = self.op_stack.pop().unwrap();
                match op.as_str() {
                    "+" => {
                        let operand1 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        let operand2 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        self.operand_stack.push((operand1 + operand2).into());
                    }
                    "-" => {
                        let operand1 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        let operand2 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        self.operand_stack.push((operand2 - operand1).into());
                    }
                    "*" => {
                        let operand1 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        let operand2 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        self.operand_stack.push((operand1 * operand2).into());
                    }
                    "/" | "div" => {
                        let operand1 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        let operand2 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        self.operand_stack.push((operand2 / operand1).into());
                    }
                    "=" => {
                        let operand1 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        let operand2 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        self.operand_stack.push((operand1._eq(operand2)).into());
                    }
                    "!=" => {
                        let operand1 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        let operand2 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        self.operand_stack.push((operand1._neq(operand2)).into());
                    }
                    ">" => {
                        let operand1 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        let operand2 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        self.operand_stack.push((operand2.gt(operand1)).into());
                    }
                    "<" => {
                        let operand1 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        let operand2 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        self.operand_stack.push((operand2.lt(operand1)).into());
                    }
                    ">=" => {
                        let operand1 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        let operand2 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        self.operand_stack.push((operand2.ge(operand1)).into());
                    }
                    "<=" => {
                        let operand1 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        let operand2 = Int::from_dynamic(self.operand_stack.pop().unwrap());
                        self.operand_stack.push((operand2.le(operand1)).into());
                    }
                    _ => panic!("Unknown op: {}", op),
                }
            } else {
                if let Some(int_token) = token.parse::<i64>().ok() {
                    if int_token < 0 {
                        self.operand_stack
                            .push((Int::from(0) - Int::from(-int_token)).into());
                    } else {
                        self.operand_stack.push(Int::from(int_token).into());
                    }
                } else if token == "true" {
                    self.operand_stack.push(Bool::from(true).into());
                } else if token == "false" {
                    self.operand_stack.push(Bool::from(false).into());
                } else {
                    panic!("Unknown token: {}", token);
                }
            }
        }
        assert!(self.op_stack.len() == 0);
        assert!(self.operand_stack.len() == 1);
        self.operand_stack.pop()
    }
}

pub fn load_parse_add(path: &PathBuf, manager: SymValManagerRef) {
    let content =
        std::fs::read_to_string(path).expect("Something went wrong reading the model file");
    let mut leaf_names = HashSet::new();
    let mut assertions = vec![];
    let t1 = Instant::now();
    for (line_idx, line) in content.lines().enumerate() {
        let mut parser = SmtlibParser::new(line);
        let parsed = parser.parse_dynamic().unwrap();
        leaf_names.extend(parser.leaf_names);
        assertions.push(parsed);
        println!("<scalar-cond>[{}]={:?}", line_idx, parsed.to_string());
    }
    let t2 = Instant::now();
    let mut manager = manager.try_write().unwrap();
    for leaf_name in &leaf_names {
        manager
            .solver
            .run_command(&Command::DeclareConst(
                Symbol(leaf_name.clone()),
                Int::sort(),
            ))
            .unwrap();
    }
    let t3 = Instant::now();
    for leaf_name in &leaf_names {
        let int = make_clean_int(leaf_name);
        manager.add_with_name(int, leaf_name.clone(), None);
    }
    let t4 = Instant::now();
    for assertion in assertions {
        let cond: Term = assertion.into();
        let cond: Bool = cond.into();
        manager.solver.assert(cond).unwrap();
    }
    let t5 = Instant::now();
    println!(
        "In load_parse_add, parse in {:?}, add_with_name in {:?}, declare const in {:?}, add assertion in {:?}",
        t2 - t1,
        t3 - t2,
        t4 - t3,
        t5 - t4,
    );
}
