from py_mtlf.core.nwdaf_context import NwdafContext, NwdafContextClient


def context_client(
    *,
    nf_instance_id: str = "11111111-1111-4111-8111-111111111111",
    api_root: str = "http://go.example",
    internal_api_root: str = "http://go-internal.example",
) -> NwdafContextClient:
    return NwdafContextClient(
        internal_api_root,
        30,
        initial=NwdafContext(
            nf_instance_id=nf_instance_id,
            api_root=api_root,
            internal_api_root=internal_api_root,
        ),
    )
