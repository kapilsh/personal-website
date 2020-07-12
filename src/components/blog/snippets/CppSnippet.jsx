import React from "react";
import { CodeBlock, dracula } from "react-code-blocks";

export const CppSnippet = (props) => {
  return (
    <div
      style={{
        fontFamily: "Source Code Pro",
      }}
    >
      <CodeBlock
        text={props.text}
        language={"cpp"}
        showLineNumbers={true}
        theme={dracula}
        wrapLines
      />
      <br />
    </div>
  );
};
