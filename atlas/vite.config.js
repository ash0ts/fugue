import { defineConfig } from "vite";

export default defineConfig({
  base: "/fugue/",
  publicDir: "public",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    modulePreload: { polyfill: false },
    rollupOptions: {
      input: {
        home: new URL("./index.html", import.meta.url).pathname,
        studies: new URL("./experiments.html", import.meta.url).pathname,
        inventory: new URL("./inventory.html", import.meta.url).pathname,
        experiment: new URL("./experiment.html", import.meta.url).pathname,
        compare: new URL("./compare.html", import.meta.url).pathname,
        methods: new URL("./methods.html", import.meta.url).pathname
      }
    }
  }
});
