import React from "react";
import { CodeBlock, dracula } from "react-code-blocks";

export const BashSnippet = (props) => {
  return (
    <div
      style={{
        fontFamily: "Source Code Pro",
      }}
    >
      <CodeBlock
        text={props.text}
        language={"bash"}
        showLineNumbers={!props.hideLineNumbers}
        theme={dracula}
        wrapLines
      />
    </div>
  );
};
