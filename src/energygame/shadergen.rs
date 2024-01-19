use std::borrow::Cow;
use std::sync::OnceLock;

use regex::{Regex, Captures};
use rustc_hash::FxHashMap;
use wgpu::ShaderSource;

use crate::energy::EnergyConf;


#[derive(Default)]
pub(super) struct ShaderPreproc<'r> {
    replacements: FxHashMap<&'r str, String>
}

impl<'r> ShaderPreproc<'r> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn define(&mut self, name: &'r str, replacement: String) {
        self.replacements.insert(name, replacement);
    }

    pub fn preprocess<'i>(&self, input: &'i str) -> Cow<'i, str> {
        let regex = {
            static ONCE: OnceLock<Regex> = OnceLock::new();
            ONCE.get_or_init(|| Regex::new(r"\$(?:(?:\{(?<braced>[[:word:]]+)\})|(?<name>[[:word:]]+))")
                .expect("Static regex"))
        };
        regex.replace_all(input, |caps: &Captures| {
            let key = caps.name("name").or_else(|| caps.name("braced")).unwrap().as_str();
            match self.replacements.get(key) {
                Some(replacement) => replacement.as_str(),
                None => "/* Missing replacement */",
            }
        })
    }

    pub fn preprocess_dump<'i>(&self, input: &'i str, name: &str) -> Cow<'i, str> {
        let processed = self.preprocess(input);
        if std::env::var_os("GPUEQUIV_DUMP").is_some() {
            let dir = "shaders_dump";
            let path = format!("{dir}/{name}_PP.wgsl");
            if let Err(e) = std::fs::create_dir_all(dir)
                .and_then(|()| std::fs::write(&path, processed.as_bytes()))
            {
                log::error!("Could not write processed shader to {}: {}", path, e);
            };
        }
        processed
    }
}

fn impl_less_eq(conf: EnergyConf) -> String {
    let mut out = "return (\n".to_string();
    let mut shift = 0;
    let mut word = 0;
    for i in 0..conf.elements {
        if i > 0 {
            out.push_str(" &&\n");
        }
        let mask = conf.energy_mask() << shift;
        out.push_str(&format!("        (a[{word}u] & 0x{mask:x}u) <= (b[{word}u] & 0x{mask:x}u)"));
        shift += conf.energy_bits();
        if shift >= u32::BITS {
            shift = 0;
            word += 1;
        }
    }
    out.push_str("\n    )");
    out
}

fn impl_eq(conf: EnergyConf) -> String {
    let mut out = "return (\n".to_string();
    for i in 0..conf.energy_size() {
        if i > 0 {
            out.push_str(" &&\n");
        }
        out.push_str(&format!("        (a[{i}u] == b[{i}u])"));
    }
    out.push_str("\n    )");
    out
}

fn unpack(var: &str, n: u32, mask: u32, bits: u32) -> String {
    let mut out = format!("array<u32,{n}u>(\n");
    let mut shift = 0;
    let mut word = 0;
    for _ in 0..n {
        out.push_str(&format!("        ({var}[{word}u] >> {shift}u) & 0x{mask:x}u,\n"));
        shift += bits;
        if shift >= u32::BITS {
            shift = 0;
            word += 1;
        }
    }
    out.push_str("    )");
    out
}

fn update1(conf: EnergyConf) -> String {
    let mut out = String::new();
    for i in 0..conf.elements {
        // 1 encodes 1-updates (decrement)
        out.push_str(&format!("    energy[{i}u] = min(energy[{i}u] + u32(updates[{i}u] == 1u), {max}u);\n",
            max=conf.max));
    }
    out
}

fn update_min(conf: EnergyConf) -> String {
    // Look for min-updates
    // 0 means no update, 1 means 1-update, everything else represents
    // the second component in the min-operation, the first being the
    // current position i. To make place for the 2 special values, 2
    // must be subtracted here.
    let mut out = "var upd = 0u;\n".to_string();
    for i in 0..conf.elements {
        out.push_str(&format!("    upd = updates[{i}u];\n"));
        out.push_str(         "    if upd > 1u {\n");
        out.push_str(         "        upd -= 2u;\n");
        out.push_str(&format!("        energy[upd] = max(energy[upd], energy[{i}u]);\n"));
        out.push_str(         "    }\n");
    }
    out
}

fn pack_energy(conf: EnergyConf, var: &str) -> String {
    let mut out = format!("array<u32,{}u>(", conf.energy_size());
    let mut i = 0;
    for _word in 0..conf.energy_size() {
        for shift in (0..u32::BITS).step_by(conf.energy_bits() as usize) {
            if shift == 0 {
                out.push_str(&format!("\n        {var}[{i}u]"));
            } else {
                out.push_str(&format!("\n            | ({var}[{i}u] << {shift}u)"));
            }
            i += 1;
            if i >= conf.elements {
                break;
            }
        }
        out.push(',');
    }
    out.push_str("\n    )");
    out
}

fn max_supremum(conf: EnergyConf) -> String {
    let mut out = String::new();
    for i in 0..conf.elements {
        out.push_str(&format!("    supremum[{i}u] = max(supremum[{i}u], energy[{i}u]);\n"));
    }
    out
}

pub(super) fn make_replacements(conf: EnergyConf) -> ShaderPreproc<'static> {
    let mut preproc = ShaderPreproc::new();

    let elements = conf.elements.to_string();
    let e_size = conf.energy_size().to_string();
    let u_size = conf.update_size().to_string();
    let less_eq = impl_less_eq(conf);
    let eq = impl_eq(conf);
    let up_energy = unpack("e", conf.elements, conf.energy_mask(), conf.energy_bits());
    let up_update = unpack("u", conf.elements, conf.update_mask(), conf.update_bits());
    let upd1 = update1(conf);
    let upd_min = update_min(conf);
    let e_pack = pack_energy(conf, "energy");
    let s_pack = pack_energy(conf, "supremum");
    let max = max_supremum(conf);

    preproc.define("ENERGY_ELEMENTS", elements);
    preproc.define("ENERGY_SIZE", e_size);
    preproc.define("UPDATE_SIZE", u_size);
    preproc.define("IMPL_LESS_EQ", less_eq);
    preproc.define("IMPL_EQ", eq);
    preproc.define("UNPACK_ENERGY", up_energy);
    preproc.define("UNPACK_UPDATE", up_update);
    preproc.define("UPDATE1", upd1);
    preproc.define("UPDATE_MIN", upd_min);
    preproc.define("PACK_ENERGY", e_pack);
    preproc.define("PACK_SUPREMUM", s_pack);
    preproc.define("MAX_SUPREMUM", max);

    preproc
}

pub(super) fn build_attack(preproc: &ShaderPreproc) -> ShaderSource<'static> {
    let shader_in = include_str!("attack.wgsl");
    ShaderSource::Wgsl(preproc.preprocess_dump(shader_in, "attack"))
}

pub(super) fn build_defend_update(preproc: &ShaderPreproc) -> ShaderSource<'static> {
    let shader_in = include_str!("defend_update.wgsl");
    ShaderSource::Wgsl(preproc.preprocess_dump(shader_in, "defend_update"))
}

pub(super) fn build_defend_intersection(preproc: &ShaderPreproc) -> ShaderSource<'static> {
    let shader_in = include_str!("defend_intersection.wgsl");
    ShaderSource::Wgsl(preproc.preprocess_dump(shader_in, "defend_intersection"))
}
