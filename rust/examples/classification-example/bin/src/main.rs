use anyhow::Result;
use std::path::Path;
use wasi_common::sync::Dir;
use wasmtime::{Config, Engine, Store};
use wasmtime_wasi::WasiCtx;
use wasmtime_wasi::{DirPerms, FilePerms, WasiCtxBuilder};
use wasmtime_wasi::{ResourceTable, WasiView};
use wasmtime_wasi_nn::{backend, Backend, InMemoryRegistry, WasiNnCtx};

use wasmtime::component::{Component, Linker as ComponentLinker};

fn main() -> wasmtime::Result<()> {
    let module_path = Path::new("target/wasm32-wasi/release/wasi_nn_example_lib.wasm");
    let preopen_dir = Path::new("build");
    let mut config = Config::new();
    config.wasm_component_model(true);

    let engine = Engine::new(&config)?;
    let backend = Backend::from(backend::openvino::OpenvinoBackend::default());
    let context = Ctx::new(preopen_dir, backend)?;

    let component = Component::from_file(&engine, &module_path).unwrap();

    let mut component_linker = ComponentLinker::new(&engine);
    wasmtime_wasi_nn::wit::ML::add_to_linker(&mut component_linker, |c: &mut Ctx| &mut c.wasi_nn)?;
    wasmtime_wasi::command::sync::add_to_linker(&mut component_linker).unwrap();

    let mut store = Store::new(&engine, context);
    let instance = component_linker.instantiate(&mut store, &component)?;
    let run_classification = {
        let mut exports = instance.exports(&mut store);
        exports.root().func("run-classification").unwrap()
    };
    run_classification.call(&mut store, &[], &mut [])?;
    Ok(())
}

struct Ctx {
    table: ResourceTable,
    wasi: WasiCtx,
    wasi_nn: WasiNnCtx,
}

impl WasiView for Ctx {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.table
    }
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.wasi
    }
}

impl Ctx {
    fn new(preopen_dir: &Path, backend: Backend) -> Result<Self> {
        // Create the WASI context.
        let preopen_dir = Dir::open_ambient_dir(preopen_dir, cap_std::ambient_authority())?;
        let wasi = WasiCtxBuilder::new()
            .inherit_stdio()
            .preopened_dir(preopen_dir, DirPerms::all(), FilePerms::all(), "build")
            .build();

        let registry = InMemoryRegistry::new();
        let wasi_nn = WasiNnCtx::new([backend.into()], registry.into());

        Ok(Self {
            table: ResourceTable::new(),
            wasi,
            wasi_nn,
        })
    }
}
