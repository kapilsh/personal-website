import React from "react";
import { CodeBlock, dracula } from "react-code-blocks";

export const PythonSnippet = (props) => {
  return (
    <div
      style={{
        fontFamily: "Source Code Pro",
      }}
    >
      <CodeBlock
        text={props.text}
        language={"python"}
        showLineNumbers={!props.hideLineNumbers}
        theme={dracula}
        wrapLines
      />
      <br />
    </div>
  );
};
