import React from "react";
import { CodeBlock, dracula } from "react-code-blocks";

export const HTMLSnippet = (props) => {
    return (
        <div
            style={{
                fontFamily: "Source Code Pro",
            }}
        >
            <CodeBlock
                text={props.text}
                language={"html"}
                showLineNumbers={false}
                theme={dracula}
                wrapLines
            />
            <br/>
        </div>
    );
};
