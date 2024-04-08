declare namespace IndexScssNamespace {
    export interface IIndexScss {
      b: string;
      container: string;
      content: string;
      figure: string;
      horizontal: string;
      indicator: string;
      l: string;
      node: string;
      nodes: string;
      r: string;
      t: string;
      timeline: string;
      title: string;
      vertical: string;
    }
  }
  
  declare const IndexScssModule: IndexScssNamespace.IIndexScss & {
    /** WARNING: Only available when `css-loader` is used without `style-loader` or `mini-css-extract-plugin` */
    locals: IndexScssNamespace.IIndexScss;
  };
  
  export = IndexScssModule;
  